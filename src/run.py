import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import openai
import torch
from tqdm import tqdm

from . import utils
from .load import Candidate, MainEntity, NeighborGenerator, build_entity_example, load_image_descriptions
from .loss import ranking_loss
from .model import (
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    AgentOutput,
    EntityState,
    FastLLMOrchestrator,
    MAORMasRouterReasoner,
    TrainableAggregator,
    build_reasoner,
)
from .utils import ensure_dir, evaluate_alignment, resolve_torch_device, save_json_atomic, transform_time


API_KEY = ""
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ROUTER_CKPT = "checkpoints/router_policy.pt"


@dataclass(frozen=True)
class DatasetDefaults:
    data_dir: str
    data: str
    image_desc_file: str
    agg_ckpt: str
    result_name: str


DATASETS = {
    "dbp": DatasetDefaults(
        "data/DBP15K", "ja_en", "ja_en_img.json",
        "checkpoints/ja_en_aggregators.pt", "ja_en_agents_stru",
    ),
    "maea": DatasetDefaults(
        "data", "icews_wiki", "WIKI_img.json",
        "checkpoints/icwiki_aggregators.pt", "icews_wiki_agents_ablation_stru",
    ),
}


class FastAggregatorTrainer:
    def __init__(
        self,
        reasoner: "MAORMasRouterReasoner",
        aggregator_model: TrainableAggregator,
        device: str = "cpu",
        agg_lr: float = 1e-3,
        margin: float = 0.1,
        lambda_margin: float = 0.5,
    ):
        self.reasoner = reasoner
        self.aggregator_model = aggregator_model
        self.device = device
        if getattr(self.aggregator_model, "use_equal_weights", False):
            print("Warning: disabling equal aggregator weights during training so fusion weights are learnable.")
            self.aggregator_model.use_equal_weights = False
        self.agg_optim = torch.optim.Adam(self.aggregator_model.parameters(), lr=agg_lr)
        self.margin = margin
        self.lambda_margin = lambda_margin

    def train_step(self, main_entity: MainEntity, candidates: List[Candidate], ref_ent: int) -> Dict[str, float]:
        state = EntityState(main_entity=main_entity, candidates=candidates)
        state = run_training_all_agents(state, self.reasoner.orchestrator)
        return self.train_on_state(state, ref_ent)

    def _target_idx(self, state: EntityState, ref_ent: int) -> int:
        for idx, cand in enumerate(state.candidates):
            if cand.ent_id == ref_ent:
                return idx
        return 0

    def _metrics_from_logits(
        self,
        state: EntityState,
        logits: torch.Tensor,
        ref_ent: int,
        agg_loss: float,
        ce_loss: float,
        margin_loss: float,
    ) -> Dict[str, float]:
        ranked_indices = torch.argsort(logits.detach(), descending=True).tolist()
        ranked_candidates = [state.candidates[idx] for idx in ranked_indices]

        top1_correct = 1.0 if ranked_candidates and ranked_candidates[0].ent_id == ref_ent else 0.0
        mrr = 0.0
        ref_rank = float(len(ranked_candidates))
        for idx, cand in enumerate(ranked_candidates, start=1):
            if cand.ent_id == ref_ent:
                mrr = 1.0 / idx
                ref_rank = float(idx - 1)
                break

        top_pred = int(ranked_candidates[0].ent_id) if ranked_candidates else -1
        return {
            "top1": float(top1_correct),
            "mrr": float(mrr),
            "ref_rank": float(ref_rank),
            "agg_loss": float(agg_loss),
            "ce_loss": float(ce_loss),
            "margin_loss": float(margin_loss),
            "token_cost": float(state.total_token_cost),
            "latency": float(state.total_latency),
            "selected_agents": ",".join(state.selected_agents),
            "used_visual_text": float("VisualTextAgent" in state.selected_agents),
            "used_image": float("ImageAgent" in state.selected_agents),
            "agent_weights": ",".join(
                f"{name}={weight:.3f}"
                for name, weight in self.aggregator_model.current_weights().items()
            ),
            "selected_weights": ",".join(
                f"{name}={weight:.3f}"
                for name, weight in self.aggregator_model.selected_weights(state.selected_agents).items()
            ),
            "top_pred": top_pred,
            "ref_ent": int(ref_ent),
            "executed": ";".join(
                f"{stage['agent']}@{stage['top_k']}#{stage['output_count']}"
                for stage in state.orchestration_trace.get("executed_stages", [])
            ),
            "stage1": ";".join(
                f"{stage['agent']}@{stage['top_k']}#{stage['output_count']}"
                for stage in state.orchestration_trace.get("executed_stages", [])
                if stage.get("stage") == "single"
            ),
            "stage2": "",
            "stage1_plan": ",".join(
                f"{stage['agent']}@{stage['top_k']}"
                for stage in state.orchestration_trace.get("stages", [])
            ),
            "stage2_plan": "",
        }

    def train_on_state(self, state: EntityState, ref_ent: int) -> Dict[str, float]:
        target_idx = self._target_idx(state, ref_ent)
        logits = self.reasoner.aggregator.training_scores(state)
        total_loss, ce_loss, margin_loss = ranking_loss(logits, target_idx, self.margin, self.lambda_margin)

        self.agg_optim.zero_grad()
        total_loss.backward()
        self.agg_optim.step()
        return self._metrics_from_logits(
            state,
            logits,
            ref_ent,
            float(total_loss.detach().cpu().item()),
            float(ce_loss.detach().cpu().item()),
            float(margin_loss.detach().cpu().item()),
        )

    def train_batch_on_states(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch_items:
            raise ValueError("batch_items must not be empty")

        batch_losses: List[torch.Tensor] = []
        batch_ce_losses: List[float] = []
        batch_margin_losses: List[float] = []
        batch_infos: List[Dict[str, Any]] = []
        for item in batch_items:
            state = item["state"]
            ref_ent = int(item["ref_ent"])
            target_idx = self._target_idx(state, ref_ent)
            logits = self.reasoner.aggregator.training_scores(state)
            total_loss, ce_loss, margin_loss = ranking_loss(logits, target_idx, self.margin, self.lambda_margin)
            batch_losses.append(total_loss)
            batch_ce_losses.append(float(ce_loss.detach().cpu().item()))
            batch_margin_losses.append(float(margin_loss.detach().cpu().item()))
            info = self._metrics_from_logits(
                state,
                logits,
                ref_ent,
                float(total_loss.detach().cpu().item()),
                float(ce_loss.detach().cpu().item()),
                float(margin_loss.detach().cpu().item()),
            )
            info["base_rank"] = int(item.get("base_rank", -1))
            batch_infos.append(info)

        mean_loss = torch.stack(batch_losses).mean()
        if not mean_loss.requires_grad:
            raise RuntimeError(
                "Aggregator loss does not require grad. Check that TrainableAggregator is not using "
                "equal fixed weights and that training_scores() calls the model without torch.no_grad()."
            )
        self.agg_optim.zero_grad()
        mean_loss.backward()
        self.agg_optim.step()

        avg_top1 = sum(info["top1"] for info in batch_infos) / len(batch_infos)
        avg_mrr = sum(info["mrr"] for info in batch_infos) / len(batch_infos)
        return {
            "batch_size": len(batch_items),
            "agg_loss": float(mean_loss.detach().cpu().item()),
            "ce_loss": sum(batch_ce_losses) / len(batch_ce_losses),
            "margin_loss": sum(batch_margin_losses) / len(batch_margin_losses),
            "top1": float(avg_top1),
            "mrr": float(avg_mrr),
            "items": batch_infos,
        }


def build_training_stages(orchestrator: FastLLMOrchestrator, state: EntityState) -> List[Dict[str, Any]]:
    stages = [
        {"agent": "TextAgent", "top_k": min(orchestrator.default_text_top_k, len(state.candidates), orchestrator.max_candidates)},
        {"agent": "StructureAgent", "top_k": min(orchestrator.default_structure_top_k, len(state.candidates), orchestrator.max_candidates)},
    ]
    has_main_visual_text = bool(state.main_entity.img_desc)
    has_candidate_visual_text = any(cand.img_desc for cand in state.candidates[:orchestrator.max_candidates])
    if has_main_visual_text and has_candidate_visual_text:
        stages.append({
            "agent": "VisualTextAgent",
            "top_k": min(orchestrator.default_visual_text_top_k, len(state.candidates), orchestrator.max_candidates),
        })

    has_main_raw_image = bool(state.main_entity.image_path)
    has_candidate_raw_image = any(cand.image_path for cand in state.candidates[: min(orchestrator.default_image_top_k, orchestrator.max_candidates)])
    if has_main_raw_image and has_candidate_raw_image:
        stages.append({
            "agent": "ImageAgent",
            "top_k": min(orchestrator.default_image_top_k, len(state.candidates), orchestrator.max_candidates),
        })
    return stages


def run_training_all_agents(state: EntityState, orchestrator: FastLLMOrchestrator) -> EntityState:
    stages = build_training_stages(orchestrator, state)
    route = {
        "text_model": orchestrator.text_model_pool[0],
        "vision_model": orchestrator.vision_model_pool[0],
        "stages": stages,
        "reason": "training all available agents",
        "used_fallback": False,
        "training_mode": True,
        "orchestrator_token_total": 0,
        "orchestrator_latency": 0.0,
    }
    state.orchestration_trace = route

    agents = orchestrator.agents_factory(text_model=route["text_model"], vision_model=route["vision_model"])
    selected_agents = []
    execution_trace = []
    for stage in stages:
        agent_name = stage["agent"]
        if agent_name not in agents:
            continue
        selected_agents.append(agent_name)
        stage_candidates = state.candidates[: int(stage["top_k"])]
        outputs, cost = agents[agent_name].run_listwise(state, stage_candidates)
        for out in outputs:
            state.agent_outputs.setdefault(out.candidate_id, []).append(out)
        state.total_token_cost += int(cost.get("token_total", 0) or 0)
        state.total_latency += float(cost.get("latency", 0.0) or 0.0)
        execution_trace.append({
            "agent": agent_name,
            "top_k": int(stage["top_k"]),
            "output_count": len(outputs),
            "token_total": int(cost.get("token_total", 0) or 0),
            "latency": float(cost.get("latency", 0.0) or 0.0),
            "stage": "single",
            "sample_scores": [
                {
                    "candidate_id": out.candidate_id,
                    "score": round(float(out.score), 4),
                    "confidence": round(float(out.confidence), 4),
                }
                for out in outputs[:3]
            ],
        })

    state.selected_agents = selected_agents
    state.orchestration_trace["executed_stages"] = execution_trace
    state.orchestration_trace["used_visual_text_agent"] = "VisualTextAgent" in selected_agents
    state.orchestration_trace["used_image_agent"] = "ImageAgent" in selected_agents
    state.orchestration_trace["used_visual_agent"] = bool({"VisualTextAgent", "ImageAgent"} & set(selected_agents))
    return state


def apply_training_agent_dropout(state: EntityState, dropout_prob: float) -> EntityState:
    if dropout_prob <= 0.0 or len(state.selected_agents) <= 1:
        return state

    kept_agents = [agent_name for agent_name in state.selected_agents if random.random() >= dropout_prob]
    if not kept_agents:
        kept_agents = [random.choice(state.selected_agents)]

    if len(kept_agents) == len(state.selected_agents):
        return state

    keep_set = set(kept_agents)
    dropped_agents = [agent_name for agent_name in state.selected_agents if agent_name not in keep_set]

    filtered_outputs: Dict[int, List[AgentOutput]] = {}
    for cand_id, outputs in state.agent_outputs.items():
        kept_outputs = [out for out in outputs if out.agent_name in keep_set]
        if kept_outputs:
            filtered_outputs[cand_id] = kept_outputs
    state.agent_outputs = filtered_outputs
    state.selected_agents = kept_agents

    if "stages" in state.orchestration_trace:
        state.orchestration_trace["stages"] = [
            stage for stage in state.orchestration_trace["stages"]
            if stage.get("agent") in keep_set
        ]
    if "executed_stages" in state.orchestration_trace:
        state.orchestration_trace["executed_stages"] = [
            stage for stage in state.orchestration_trace["executed_stages"]
            if stage.get("agent") in keep_set
        ]
    state.orchestration_trace["used_visual_text_agent"] = "VisualTextAgent" in keep_set
    state.orchestration_trace["used_image_agent"] = "ImageAgent" in keep_set
    state.orchestration_trace["used_visual_agent"] = bool({"VisualTextAgent", "ImageAgent"} & keep_set)
    state.orchestration_trace["training_dropout_prob"] = dropout_prob
    state.orchestration_trace["dropped_agents"] = dropped_agents
    return state


def prepare_training_state(
    ng: NeighborGenerator,
    ent_id: int,
    dataset_name: str,
    image_root: Optional[str],
    image_desc: Optional[Dict[int, str]],
    reasoner: MAORMasRouterReasoner,
    apply_agent_dropout: bool = False,
    agent_dropout_prob: float = 0.0,
) -> Dict[str, Any]:
    example = build_entity_example(ng, ent_id, dataset_name, image_root, image_desc)
    state = EntityState(main_entity=example["main_entity"], candidates=example["candidates"])
    state = run_training_all_agents(state, reasoner.orchestrator)
    if apply_agent_dropout:
        state = apply_training_agent_dropout(state, agent_dropout_prob)
    return {
        "ent_id": int(example["ent_id"]),
        "ref_ent": int(example["ref_ent"]),
        "base_rank": int(example["base_rank"]),
        "state": state,
    }


def train_router_and_aggregator(
    data: str,
    data_dir: str,
    cand_file: str,
    client: openai.OpenAI,
    image_root: Optional[str],
    image_desc_path: Optional[str],
    text_model_pool: List[str],
    vision_model_pool: List[str],
    router_ckpt: str,
    agg_ckpt: str,
    train_epochs: int = 20,
    train_sample_size: int = 100,
    train_batch_size: int = 20,
    train_workers: int = 4,
    train_rank_min: int = 2,
    train_rank_max: int = 20,
    train_seed: int = 42,
    dropout_start_epoch: int = 11,
    agent_dropout_prob: float = 0.2,
    margin: float = 0.1,
    lambda_margin: float = 0.5,
    save_every: int = 20,
    device: str = "cpu",
    *,
    vision_client: Optional[openai.OpenAI] = None,
) -> None:
    device = resolve_torch_device(device)
    print(f"Training device: {device}")
    image_desc = load_image_descriptions(image_desc_path)
    print(f"Loaded image descriptions: {len(image_desc)}")
    ng = NeighborGenerator(data=data, data_dir=data_dir, cand_file=cand_file)
    reasoner = build_reasoner(
        client, text_model_pool, vision_model_pool, agg_ckpt, device, vision_client=vision_client
    )
    aggregator_model = reasoner.aggregator.model
    trainer = FastAggregatorTrainer(
        reasoner,
        aggregator_model,
        device=device,
        margin=margin,
        lambda_margin=lambda_margin,
    )
    print("Router policy training is disabled. Training runs all available agents, then updates only the aggregator.")

    all_ids = list(ng.get_entities())
    if not all_ids:
        print("No entities found.")
        return

    eligible_ids = [
        ent_id for ent_id in all_ids
    ]
    if not eligible_ids:
        print("No eligible entities found for training.")
        return

    print(
        f"Eligible training entities: {len(eligible_ids)} "
        f"(base_rank in [{train_rank_min}, {train_rank_max}])"
    )
    rng = random.Random(train_seed)
    sample_size = min(max(1, train_sample_size), len(eligible_ids))
    batch_size = max(1, train_batch_size)
    fixed_train_ids = rng.sample(eligible_ids, k=sample_size)
    print(
        f"Fixed training entities: {len(fixed_train_ids)} "
        f"(seed={train_seed})"
    )
    global_step = 0

    for epoch in range(1, train_epochs + 1):
        epoch_ids = fixed_train_ids[:]
        rng.shuffle(epoch_ids)
        epoch_stats: List[Dict[str, Any]] = []
        prepared_items: List[Dict[str, Any]] = []
        worker_count = max(1, train_workers)
        apply_dropout = epoch >= dropout_start_epoch
        phase_name = "dropout" if apply_dropout else "full"

        if worker_count <= 1:
            for ent_id in tqdm(epoch_ids, desc=f"Prepare Epoch {epoch}/{train_epochs} ({phase_name})"):
                prepared_items.append(
                    prepare_training_state(
                        ng,
                        ent_id,
                        data,
                        image_root,
                        image_desc,
                        reasoner,
                        apply_agent_dropout=apply_dropout,
                        agent_dropout_prob=agent_dropout_prob,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        prepare_training_state,
                        ng,
                        ent_id,
                        data,
                        image_root,
                        image_desc,
                        reasoner,
                        apply_dropout,
                        agent_dropout_prob,
                    ): ent_id
                    for ent_id in epoch_ids
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Prepare Epoch {epoch}/{train_epochs} ({phase_name})"):
                    prepared_items.append(future.result())

        batch_count = (len(prepared_items) + batch_size - 1) // batch_size
        train_pbar = tqdm(range(batch_count), desc=f"Train Epoch {epoch}/{train_epochs} ({phase_name})")
        for batch_idx in train_pbar:
            batch_items = prepared_items[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_info = trainer.train_batch_on_states(batch_items)
            epoch_stats.extend(batch_info["items"])
            global_step += 1
            train_pbar.set_postfix(
                loss=f"{batch_info['agg_loss']:.4f}",
                ce=f"{batch_info['ce_loss']:.4f}",
                margin=f"{batch_info['margin_loss']:.4f}",
                top1=f"{batch_info['top1']:.4f}",
                mrr=f"{batch_info['mrr']:.4f}",
            )

            print(
                f"[epoch {epoch}/{train_epochs} {phase_name} batch {batch_idx + 1}/{batch_count} step {global_step}] "
                f"batch_size={batch_info['batch_size']} "
                f"loss={batch_info['agg_loss']:.4f} "
                f"ce={batch_info['ce_loss']:.4f} "
                f"margin={batch_info['margin_loss']:.4f} "
                f"top1={batch_info['top1']:.4f} "
                f"mrr={batch_info['mrr']:.4f}"
            )
            if save_every > 0 and global_step % save_every == 0:
                ensure_dir(os.path.dirname(agg_ckpt) or ".")
                torch.save(aggregator_model.state_dict(), agg_ckpt)

        if epoch_stats:
            avg_top1 = sum(item["top1"] for item in epoch_stats) / len(epoch_stats)
            avg_mrr = sum(item["mrr"] for item in epoch_stats) / len(epoch_stats)
            avg_loss = sum(item["agg_loss"] for item in epoch_stats) / len(epoch_stats)
            avg_ce = sum(item["ce_loss"] for item in epoch_stats) / len(epoch_stats)
            avg_margin = sum(item["margin_loss"] for item in epoch_stats) / len(epoch_stats)
            print(
                f"[epoch {epoch}/{train_epochs} {phase_name} summary] "
                f"sample_size={len(epoch_stats)} top1={avg_top1:.4f} "
                f"mrr={avg_mrr:.4f} loss={avg_loss:.4f} "
                f"ce={avg_ce:.4f} margin={avg_margin:.4f}"
            )

    ensure_dir(os.path.dirname(agg_ckpt) or ".")
    torch.save(aggregator_model.state_dict(), agg_ckpt)
    print(f"saved aggregator -> {agg_ckpt}")


def rerank_one_entity(reasoner: MAORMasRouterReasoner, main_entity: MainEntity, candidates: List[Candidate], ref_ent: int) -> Dict[str, Any]:
    st = time.time()
    ranked, state = reasoner.rerank_with_state(main_entity, candidates)
    final_rank = len(ranked)
    for i, item in enumerate(ranked):
        if item["ent_id"] == ref_ent:
            final_rank = i
            break
    return {
        "ranked": ranked,
        "llm_rank": int(final_rank),
        "time_cost": time.time() - st,
        "token_cost": int(state.total_token_cost),
        "latency_cost": float(state.total_latency),
        "route": state.orchestration_trace,
        "selected_agents": state.selected_agents,
    }


def build_bypass_result(base_rank: int, candidates: List[Candidate]) -> Dict[str, Any]:
    return {
        "base_rank": int(base_rank),
        "llm_rank": int(base_rank),
        "time_cost": 0.0,
        "token_cost": 0,
        "latency_cost": 0.0,
        "selected_agents": [],
        "route": {"mode_name": "bypass"},
        "top5": [
            {
                "ent_id": c.ent_id,
                "name": c.name,
                "base_score": c.base_score,
                "final_score": c.base_score,
                "agent_outputs": [],
            }
            for c in candidates[:5]
        ],
    }


def rerank_entity_example(
    reasoner: MAORMasRouterReasoner,
    example: Dict[str, Any],
    bypass_rank_threshold: int = 29,
) -> Tuple[int, Dict[str, Any]]:
    ent_id = int(example["ent_id"])
    base_rank = int(example["base_rank"])
    candidates = example["candidates"]
    if base_rank >= bypass_rank_threshold:
        return ent_id, build_bypass_result(base_rank, candidates)

    rerank_info = rerank_one_entity(
        reasoner,
        example["main_entity"],
        candidates,
        int(example["ref_ent"]),
    )
    return ent_id, {
        "base_rank": base_rank,
        "llm_rank": int(rerank_info["llm_rank"]),
        "time_cost": float(rerank_info["time_cost"]),
        "token_cost": int(rerank_info["token_cost"]),
        "latency_cost": float(rerank_info["latency_cost"]),
        "selected_agents": rerank_info["selected_agents"],
        "route": rerank_info["route"],
        "top5": rerank_info["ranked"][:5],
    }


def re_rank_by_maor_ea(ng: NeighborGenerator, dataset_name: str, image_root: Optional[str], reasoner: MAORMasRouterReasoner, save_result_path: str, save_step: int = 10, use_previous_result: bool = False, bypass_rank_threshold: int = 29, image_desc: Optional[Dict[int, str]] = None, eval_workers: int = 4) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    if use_previous_result and os.path.exists(save_result_path):
        try:
            with open(save_result_path, "r", encoding="utf-8") as fr:
                raw = json.load(fr)
                result = {int(k): v for k, v in raw.items()}
        except Exception:
            result = {}

    pending_examples = [
        build_entity_example(ng, ent_id, dataset_name, image_root, image_desc)
        for ent_id in ng.get_entities()
        if ent_id not in result
        #if result[ent_id]['route']['reason'] == "fallback fast text+structure plan"
    ]

    ent_num = 0
    if not pending_examples:
        return result

    if eval_workers <= 1:
        for example in tqdm(pending_examples, desc="MAOR-EA MasRouter Rerank"):
            ent_id, entity_result = rerank_entity_example(reasoner, example, bypass_rank_threshold)
            result[ent_id] = entity_result
            print(result[ent_id])
            ent_num += 1
            if save_step > 0 and ent_num % save_step == 0:
                save_json_atomic(save_result_path, result)
        return result

    with ThreadPoolExecutor(max_workers=max(1, eval_workers)) as executor:
        future_to_ent = {
            executor.submit(rerank_entity_example, reasoner, example, bypass_rank_threshold): int(example["ent_id"])
            for example in pending_examples
        }
        for future in tqdm(as_completed(future_to_ent), total=len(future_to_ent), desc="MAOR-EA MasRouter Rerank"):
            ent_id, entity_result = future.result()
            result[ent_id] = entity_result
            print(result[ent_id])
            ent_num += 1
            if save_step > 0 and ent_num % save_step == 0:
                save_json_atomic(save_result_path, result)
    return result


def evaluate(
    data: str,
    data_dir: str,
    cand_file: str,
    client: openai.OpenAI,
    text_model_pool: List[str],
    vision_model_pool: List[str],
    image_root: Optional[str],
    image_desc_path: Optional[str],
    result_name: str,
    save_step: int,
    use_previous_result: bool,
    router_ckpt: str,
    agg_ckpt: str,
    device: str = "cpu",
    hit_k: Optional[List[int]] = None,
    eval_workers: int = 4,
    *,
    vision_client: Optional[openai.OpenAI] = None,
) -> Tuple[Dict[int, Dict[str, Any]], List[float], float]:
    device = resolve_torch_device(device)
    print(f"Evaluation device: {device}")
    if hit_k is None:
        hit_k = [1, 5, 10, 20, 30]
    image_desc = load_image_descriptions(image_desc_path)
    print(f"Loaded image descriptions: {len(image_desc)}")

    ng = NeighborGenerator(data=data, data_dir=data_dir, cand_file=cand_file)
    reasoner = build_reasoner(
        client, text_model_pool, vision_model_pool, agg_ckpt, device, vision_client=vision_client
    )

    save_dir = os.path.join("result", data)
    ensure_dir(save_dir)
    save_result_path = os.path.join(save_dir, f"{result_name}.json")
    main_st = time.time()

    result = re_rank_by_maor_ea(
        ng=ng,
        dataset_name=data,
        image_root=image_root,
        reasoner=reasoner,
        save_result_path=save_result_path,
        save_step=save_step,
        use_previous_result=use_previous_result,
        image_desc=image_desc,
        eval_workers=eval_workers,
    )

    base_ranks, rerank_ranks = [], []
    total_time_cost, total_token_cost = 0.0, 0
    visual_text_used, image_used = 0, 0
    for _, v in result.items():
        base_ranks.append(v["base_rank"])
        rerank_ranks.append(v["llm_rank"])
        total_time_cost += v.get("time_cost", 0.0)
        total_token_cost += int(v.get("token_cost", 0))
        route_info = v.get("route", {})
        visual_text_used += int(bool(route_info.get("used_visual_text_agent", False)))
        image_used += int(bool(route_info.get("used_image_agent", False)))

    base_hits, base_mrr = evaluate_alignment(base_ranks, hit_k=hit_k)
    rerank_hits, rerank_mrr = evaluate_alignment(rerank_ranks, hit_k=hit_k)

    print(f"Base Rank       : Hits@{hit_k} = {base_hits} , MRR = {base_mrr:.3f}")
    print(f"MAOR-MasRouter  : Hits@{hit_k} = {rerank_hits} , MRR = {rerank_mrr:.3f}")
    wall_time = time.time() - main_st
    h, m, s = transform_time(int(wall_time))
    print(f"Wall Time       : {h}hour, {m:02d}min, {s:02d}sec")
    print(f"Entity API Time : {total_time_cost:.2f}s")
    print(f"Total Tokens    : {total_token_cost}")
    total_entities = max(1, len(result))
    print(f"VisualText Used : {visual_text_used}/{len(result)} ({visual_text_used / total_entities:.2%})")
    print(f"Image Used      : {image_used}/{len(result)} ({image_used / total_entities:.2%})")

    save_json_atomic(save_result_path, result)
    return result, rerank_hits, rerank_mrr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate MAEC.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--run_mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--api_key", help="shared API key for text and vision")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="shared API base URL")
    parser.add_argument("--text_api_key")
    parser.add_argument("--text_base_url")
    parser.add_argument("--vision_api_key")
    parser.add_argument("--vision_base_url")
    parser.add_argument("--text_model_pool", type=str, default=DEFAULT_TEXT_MODEL, help="comma separated, e.g. qwen-plus,qwen-max")
    parser.add_argument("--vision_model_pool", type=str, default=DEFAULT_VISION_MODEL, help="comma separated")
    parser.add_argument("--data_dir")
    parser.add_argument("--data")
    parser.add_argument("--cand_file", type=str, default="cand")
    parser.add_argument("--trainCand_file", type=str, default="train_cand")
    parser.add_argument("--image_root", help="image directory (default: DATA_DIR/DATA/img)")
    parser.add_argument("--image_desc_path", help="image-description JSON (default: dataset-specific file)")
    parser.add_argument("--router_ckpt", type=str, default=DEFAULT_ROUTER_CKPT)
    parser.add_argument("--agg_ckpt")
    parser.add_argument("--result_name")
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--new_result", action="store_true")
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--train_sample_size", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=20)
    parser.add_argument("--train_workers", type=int, default=20)
    parser.add_argument("--train_rank_min", type=int, default=1)
    parser.add_argument("--train_rank_max", type=int, default=5)
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--dropout_start_epoch", type=int, default=11)
    parser.add_argument("--agent_dropout_prob", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--lambda_margin", type=float, default=0.5)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--eval_workers", type=int, default=100)
    parser.add_argument("--log_print", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = DATASETS[args.dataset]
    data_dir = defaults.data_dir if args.data_dir is None else args.data_dir
    data = defaults.data if args.data is None else args.data
    dataset_path = os.path.join(data_dir, data)
    image_root = args.image_root if args.image_root is not None else os.path.join(dataset_path, "img")
    image_desc_path = args.image_desc_path if args.image_desc_path is not None else os.path.join(
        dataset_path, defaults.image_desc_file
    )
    agg_ckpt = defaults.agg_ckpt if args.agg_ckpt is None else args.agg_ckpt
    result_name = defaults.result_name if args.result_name is None else args.result_name

    text_model_pool = [x.strip() for x in args.text_model_pool.split(",") if x.strip()] or [DEFAULT_TEXT_MODEL]
    vision_model_pool = [x.strip() for x in args.vision_model_pool.split(",") if x.strip()] or [DEFAULT_VISION_MODEL]
    api_key = args.api_key or API_KEY
    client = openai.OpenAI(
        api_key=args.text_api_key or api_key,
        base_url=args.text_base_url or args.base_url or DEFAULT_BASE_URL,
    )
    vision_client = openai.OpenAI(
        api_key=args.vision_api_key or api_key,
        base_url=args.vision_base_url or args.base_url or DEFAULT_BASE_URL,
    )

    utils.LOG_PRINT = args.log_print

    if args.run_mode == "train":
        train_router_and_aggregator(
            data=data,
            data_dir=data_dir,
            cand_file=args.trainCand_file,
            client=client,
            vision_client=vision_client,
            image_root=image_root,
            image_desc_path=image_desc_path,
            text_model_pool=text_model_pool,
            vision_model_pool=vision_model_pool,
            router_ckpt=args.router_ckpt,
            agg_ckpt=agg_ckpt,
            train_epochs=args.train_epochs,
            train_sample_size=args.train_sample_size,
            train_batch_size=args.train_batch_size,
            train_workers=args.train_workers,
            train_rank_min=args.train_rank_min,
            train_rank_max=args.train_rank_max,
            train_seed=args.train_seed,
            dropout_start_epoch=args.dropout_start_epoch,
            agent_dropout_prob=args.agent_dropout_prob,
            margin=args.margin,
            lambda_margin=args.lambda_margin,
            save_every=args.save_every,
            device=args.device,
        )
    else:
        evaluate(
            data=data,
            data_dir=data_dir,
            cand_file=args.cand_file,
            client=client,
            vision_client=vision_client,
            text_model_pool=text_model_pool,
            vision_model_pool=vision_model_pool,
            image_root=image_root,
            image_desc_path=image_desc_path,
            result_name=result_name,
            save_step=args.save_step,
            use_previous_result=(not args.new_result),
            router_ckpt=args.router_ckpt,
            agg_ckpt=agg_ckpt,
            device=args.device,
            hit_k=[1, 5, 10, 20, 30],
            eval_workers=args.eval_workers,
        )


if __name__ == "__main__":
    main()
