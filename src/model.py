import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import openai
import torch
import torch.nn as nn

from .load import Candidate, MainEntity
from .utils import (
    clamp,
    compute_neighbor_overlap,
    convert_neighbor_to_str,
    extract_json_payload,
    feature_matrix,
    feature_tensor,
    image_to_data_url,
    load_state_dict_if_compatible,
    name_overlap_score,
    safe_print,
    sanitize_feature_vector,
    truncate_text,
)


DEFAULT_TEXT_MODEL = "qwen-plus"
DEFAULT_VISION_MODEL = "qwen-vl-plus"


@dataclass
class AgentOutput:
    agent_name: str
    candidate_id: int
    score: float
    confidence: float
    rationale: str = ""
    token_in: int = 0
    token_out: int = 0
    token_total: int = 0
    latency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityState:
    main_entity: MainEntity
    candidates: List[Candidate]
    agent_outputs: Dict[int, List[AgentOutput]] = field(default_factory=dict)
    selected_agents: List[str] = field(default_factory=list)
    orchestration_trace: Dict[str, Any] = field(default_factory=dict)
    route_logprob: Optional[torch.Tensor] = None
    total_token_cost: int = 0
    total_latency: float = 0.0
    stop_reason: str = ""


def compact_candidate_table(main_entity: MainEntity, candidates: List[Candidate], max_candidates: int = 30) -> List[Dict[str, Any]]:
    rows = []
    for rank, cand in enumerate(candidates[:max_candidates], start=1):
        rows.append({
            "rank": rank,
            "ent_id": cand.ent_id,
            "name": truncate_text(cand.name, 80),
            "base_score": round(float(cand.base_score), 4),
            "name_overlap": round(name_overlap_score(main_entity.name, cand.name), 4),
            "neighbor_overlap": round(compute_neighbor_overlap(main_entity.neighbors, cand.neighbors), 4),
            "neighbor_count": len(cand.neighbors),
            "has_image": bool(cand.image_path),
            "has_image_desc": bool(cand.img_desc),
            "image_desc": truncate_text(cand.img_desc, 140),
        })
    return rows


def normalize_score_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("scores", "candidates", "results", "ranking"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


class ListwiseLLMAgent:
    def __init__(self, name: str, goal: str, model: str, client: openai.OpenAI, max_tokens: int = 900, temperature: float = 0.1):
        self.name = name
        self.goal = goal
        self.model = model
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature

    def build_prompt(self, state: EntityState, candidates: List[Candidate]) -> str:
        raise NotImplementedError

    def fallback_score(self, state: EntityState, candidate: Candidate) -> Tuple[float, float, str]:
        return 0.5, 0.1, "fallback"

    def call_api(self, prompt: str) -> Dict[str, Any]:
        st = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.goal},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = time.time() - st
        usage = getattr(response, "usage", None)
        return {
            "text": (response.choices[0].message.content or "").strip(),
            "token_in": int(getattr(usage, "prompt_tokens", 0) or 0),
            "token_out": int(getattr(usage, "completion_tokens", 0) or 0),
            "token_total": int(getattr(usage, "total_tokens", 0) or 0),
            "latency": latency,
        }

    def parse_outputs(self, text: str, state: EntityState, candidates: List[Candidate]) -> List[AgentOutput]:
        payload = extract_json_payload(text)
        items = normalize_score_items(payload)
        by_id: Dict[int, Dict[str, Any]] = {}
        for item in items:
            raw_id = item.get("candidate_id", item.get("ent_id", item.get("id")))
            if raw_id is None:
                continue
            try:
                by_id[int(raw_id)] = item
            except Exception:
                continue

        outputs = []
        for cand in candidates:
            item = by_id.get(cand.ent_id)
            if item is None:
                score, confidence, rationale = self.fallback_score(state, cand)
            else:
                score = clamp(float(item.get("score", item.get("alignment_score", 0.5))))
                confidence = clamp(float(item.get("confidence", 0.5)))
                rationale = truncate_text(item.get("rationale", item.get("reason", "")), 120)
            outputs.append(AgentOutput(
                agent_name=self.name,
                candidate_id=cand.ent_id,
                score=score,
                confidence=confidence,
                rationale=rationale,
            ))
        return outputs

    def run_listwise(self, state: EntityState, candidates: List[Candidate]) -> Tuple[List[AgentOutput], Dict[str, Any]]:
        if not candidates:
            return [], {"token_total": 0, "latency": 0.0}
        prompt = self.build_prompt(state, candidates)
        try:
            api_result = self.call_api(prompt)
            outputs = self.parse_outputs(api_result["text"], state, candidates)
            return outputs, api_result
        except Exception as e:
            safe_print(f"[{self.name}] listwise API error: {e}")
            outputs = []
            for cand in candidates:
                score, confidence, rationale = self.fallback_score(state, cand)
                outputs.append(AgentOutput(
                    agent_name=self.name,
                    candidate_id=cand.ent_id,
                    score=score,
                    confidence=confidence,
                    rationale=rationale,
                ))
            return outputs, {"token_total": 0, "latency": 0.0}


class ListwiseTextAgent(ListwiseLLMAgent):
    def build_prompt(self, state: EntityState, candidates: List[Candidate]) -> str:
        payload = {
            "main_entity": {
                "ent_id": state.main_entity.ent_id,
                "name": state.main_entity.name,
            },
            "candidates": [
                {
                    "rank": rank,
                    "candidate_id": cand.ent_id,
                    "name": cand.name,
                }
                for rank, cand in enumerate(candidates, start=1)
            ],
        }
        return f"""
Fast listwise entity alignment by names only.
You can only use the entity names in the input.
Do not use any original similarity scores, graph/neighborhood evidence, images, or visual descriptions.
Focus on entity identity consistency from names, including aliases, transliterations, abbreviations, and word-order variation.
Return compact JSON only:
{{"scores":[{{"candidate_id":123,"score":0.0,"confidence":0.0}}]}}

Input:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    def fallback_score(self, state: EntityState, candidate: Candidate) -> Tuple[float, float, str]:
        score = name_overlap_score(state.main_entity.name, candidate.name)
        return clamp(score), 0.25, "name-only heuristic fallback"


class ListwiseStructureAgent(ListwiseLLMAgent):
    def build_prompt(self, state: EntityState, candidates: List[Candidate]) -> str:
        rows = []
        for rank, cand in enumerate(candidates, start=1):
            rows.append({
                "rank": rank,
                "candidate_id": cand.ent_id,
                "neighbor_count": len(cand.neighbors),
                "neighbors": [convert_neighbor_to_str(x) for x in cand.neighbors[:8]],
            })
        payload = {
            "main_entity": {
                "ent_id": state.main_entity.ent_id,
                "neighbor_count": len(state.main_entity.neighbors),
                "neighbors": [convert_neighbor_to_str(x) for x in state.main_entity.neighbors[:8]],
            },
            "candidates": rows,
        }
        return f"""
Fast listwise entity alignment by graph/neighbor structure.
You can only use structural evidence: neighbor IDs, relations, and graph context.
Do not use names, textual descriptions, visual descriptions, images, or original similarity scores.
Return compact JSON only:
{{"scores":[{{"candidate_id":123,"score":0.0,"confidence":0.0}}]}}

Input:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    def fallback_score(self, state: EntityState, candidate: Candidate) -> Tuple[float, float, str]:
        score = compute_neighbor_overlap(state.main_entity.neighbors, candidate.neighbors)
        return clamp(score), 0.25, "structure heuristic fallback"


class ListwiseVisualTextAgent(ListwiseLLMAgent):
    def build_prompt(self, state: EntityState, candidates: List[Candidate]) -> str:
        rows = [
            {
                "candidate_id": cand.ent_id,
                "has_image_desc": bool(cand.img_desc),
                "image_desc": truncate_text(cand.img_desc, 180),
            }
            for cand in candidates
        ]
        return f"""
Fast listwise entity alignment by visual text descriptions only.
You can only use the textual image descriptions from img.json.
Do not use entity names, original similarity scores, graph structure, or raw images.
Return compact JSON only:
{{"scores":[{{"candidate_id":123,"score":0.0,"confidence":0.0}}]}}

Main entity: id={state.main_entity.ent_id}, image_desc={truncate_text(state.main_entity.img_desc, 240)}
Candidates:
{json.dumps(rows, ensure_ascii=False)}
""".strip()

    def run_listwise(self, state: EntityState, candidates: List[Candidate]) -> Tuple[List[AgentOutput], Dict[str, Any]]:
        if not candidates:
            return [], {"token_total": 0, "latency": 0.0}
        if not state.main_entity.img_desc:
            outputs = [
                AgentOutput("VisualTextAgent", cand.ent_id, 0.5, 0.1, "missing main visual description")
                for cand in candidates
            ]
            return outputs, {"token_total": 0, "latency": 0.0}

        visual_candidates = [cand for cand in candidates if cand.img_desc]
        if not visual_candidates:
            outputs = [
                AgentOutput("VisualTextAgent", cand.ent_id, 0.5, 0.1, "missing candidate visual description")
                for cand in candidates
            ]
            return outputs, {"token_total": 0, "latency": 0.0}

        outputs, cost = super().run_listwise(state, visual_candidates)
        output_by_id = {out.candidate_id: out for out in outputs}
        all_outputs = []
        for cand in candidates:
            if cand.ent_id in output_by_id:
                all_outputs.append(output_by_id[cand.ent_id])
            else:
                all_outputs.append(AgentOutput("VisualTextAgent", cand.ent_id, 0.5, 0.1, "missing candidate visual description"))
        return all_outputs, cost

    def fallback_score(self, state: EntityState, candidate: Candidate) -> Tuple[float, float, str]:
        return 0.5, 0.1, "visual text fallback"


class ListwiseImageAgent(ListwiseLLMAgent):
    def build_prompt(self, state: EntityState, candidates: List[Candidate]) -> str:
        rows = [
            {
                "candidate_id": cand.ent_id,
                "has_image": bool(cand.image_path),
            }
            for cand in candidates
        ]
        return f"""
Fast listwise entity alignment by raw images only.
You can only use the attached main image and candidate images.
Do not use names, original similarity scores, graph structure, or visual text descriptions.
Return compact JSON only:
{{"scores":[{{"candidate_id":123,"score":0.0,"confidence":0.0}}]}}

Main entity image id={state.main_entity.ent_id}
Candidate image IDs:
{json.dumps(rows, ensure_ascii=False)}
""".strip()

    def run_listwise(self, state: EntityState, candidates: List[Candidate]) -> Tuple[List[AgentOutput], Dict[str, Any]]:
        if not candidates:
            return [], {"token_total": 0, "latency": 0.0}
        if not state.main_entity.image_path:
            outputs = [
                AgentOutput("ImageAgent", cand.ent_id, 0.5, 0.1, "missing main raw image")
                for cand in candidates
            ]
            return outputs, {"token_total": 0, "latency": 0.0}

        image_candidates = [cand for cand in candidates if cand.image_path]
        if not image_candidates:
            outputs = [
                AgentOutput("ImageAgent", cand.ent_id, 0.5, 0.1, "missing candidate raw image")
                for cand in candidates
            ]
            return outputs, {"token_total": 0, "latency": 0.0}

        prompt = self.build_prompt(state, image_candidates)
        try:
            content = [{"type": "text", "text": prompt}]
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(state.main_entity.image_path)}})
            for cand in image_candidates:
                content.append({"type": "text", "text": f"Candidate image: candidate_id={cand.ent_id}"})
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(cand.image_path)}})

            st = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.goal},
                    {"role": "user", "content": content},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            latency = time.time() - st
            usage = getattr(response, "usage", None)
            text = (response.choices[0].message.content or "").strip()
            outputs = self.parse_outputs(text, state, image_candidates)
            output_by_id = {out.candidate_id: out for out in outputs}
            all_outputs = []
            for cand in candidates:
                if cand.ent_id in output_by_id:
                    out = output_by_id[cand.ent_id]
                    out.agent_name = "ImageAgent"
                    all_outputs.append(out)
                else:
                    all_outputs.append(AgentOutput("ImageAgent", cand.ent_id, 0.5, 0.1, "missing candidate raw image"))
            return all_outputs, {
                "token_total": int(getattr(usage, "total_tokens", 0) or 0),
                "latency": latency,
            }
        except Exception as e:
            safe_print(f"[ImageAgent] listwise API error: {e}")
            outputs = [
                AgentOutput("ImageAgent", cand.ent_id, 0.5, 0.1, "image fallback")
                for cand in candidates
            ]
            return outputs, {"token_total": 0, "latency": 0.0}

    def fallback_score(self, state: EntityState, candidate: Candidate) -> Tuple[float, float, str]:
        return 0.5, 0.1, "image fallback"


def build_listwise_agents_factory(
    client: openai.OpenAI, vision_client: Optional[openai.OpenAI] = None
):
    vision_client = vision_client if vision_client is not None else client

    def factory(text_model: str, vision_model: str) -> Dict[str, ListwiseLLMAgent]:
        return {
            "TextAgent": ListwiseTextAgent(
                "TextAgent",
                "You are a fast text-based listwise entity alignment agent.",
                text_model,
                client,
                max_tokens=900,
            ),
            "StructureAgent": ListwiseStructureAgent(
                "StructureAgent",
                "You are a fast structure-based listwise entity alignment agent.",
                text_model,
                client,
                max_tokens=900,
            ),
            "VisualTextAgent": ListwiseVisualTextAgent(
                "VisualTextAgent",
                "You are a fast visual-text entity alignment agent. Use only image descriptions.",
                vision_model,
                vision_client,
                max_tokens=900,
            ),
            "ImageAgent": ListwiseImageAgent(
                "ImageAgent",
                "You are a fast raw-image multimodal entity alignment agent. Use only images.",
                vision_model,
                vision_client,
                max_tokens=650,
            ),
        }
    return factory


class FastLLMOrchestrator:
    def __init__(
        self,
        client: openai.OpenAI,
        agents_factory,
        text_model_pool: List[str],
        vision_model_pool: List[str],
        orchestrator_model: Optional[str] = None,
        max_candidates: int = 30,
        default_text_top_k: int = 30,
        default_structure_top_k: int = 30,
        default_visual_text_top_k: int = 30,
        default_image_top_k: int = 5,
    ):
        self.client = client
        self.agents_factory = agents_factory
        self.text_model_pool = text_model_pool or [DEFAULT_TEXT_MODEL]
        self.vision_model_pool = vision_model_pool or [DEFAULT_VISION_MODEL]
        self.orchestrator_model = orchestrator_model or self.text_model_pool[-1]
        self.max_candidates = max_candidates
        self.default_text_top_k = default_text_top_k
        self.default_structure_top_k = default_structure_top_k
        self.default_visual_text_top_k = default_visual_text_top_k
        self.default_image_top_k = default_image_top_k

    def build_prompt(self, state: EntityState) -> str:
        scores = [float(c.base_score) for c in state.candidates[:self.max_candidates]]
        top1 = scores[0] if scores else 0.0
        top2 = scores[1] if len(scores) > 1 else 0.0
        visual_candidate_count = sum(
            1
            for cand in state.candidates[:self.max_candidates]
            if cand.img_desc
        )
        image_candidate_count = sum(
            1
            for cand in state.candidates[: min(5, self.max_candidates)]
            if cand.image_path
        )
        payload = {
            "goal": "Choose a fast multi-agent execution plan for top30 entity reranking.",
            "constraints": {
                "prefer_listwise_agents": True,
                "optimize_for_speed": True,
                "max_candidates": self.max_candidates,
                "single_stage_only": True,
                "allowed_agents": ["TextAgent", "StructureAgent", "VisualTextAgent", "ImageAgent"],
                "recommended_top_k": {
                    "TextAgent": self.default_text_top_k,
                    "StructureAgent": self.default_structure_top_k,
                    "VisualTextAgent": self.default_visual_text_top_k,
                    "ImageAgent": self.default_image_top_k,
                },
            },
            "main_entity": {
                "ent_id": state.main_entity.ent_id,
                "name": state.main_entity.name,
                "neighbor_count": len(state.main_entity.neighbors),
                "has_image": bool(state.main_entity.image_path),
                "has_image_desc": bool(state.main_entity.img_desc),
            },
            "score_summary": {
                "top1": round(top1, 4),
                "top2": round(top2, 4),
                "margin": round(top1 - top2, 4),
            },
            "visual_availability": {
                "main_has_image_desc": bool(state.main_entity.img_desc),
                "candidate_count_with_image_desc_top30": visual_candidate_count,
                "main_has_raw_image": bool(state.main_entity.image_path),
                "candidate_count_with_raw_image_top5": image_candidate_count,
            },
            "candidates": [
                {
                    "rank": rank,
                    "candidate_id": cand.ent_id,
                    "name": truncate_text(cand.name, 80),
                    "base_score": round(float(cand.base_score), 4),
                    "hhea_sim": round(float(cand.extra.get("hhea_sim", cand.base_score)), 4),
                    "mmea_sim": round(float(cand.extra.get("mmea_sim", cand.base_score)), 4),
                    "neighbor_count": len(cand.neighbors),
                    "has_image_desc": bool(cand.img_desc),
                    "has_raw_image": bool(cand.image_path),
                }
                for rank, cand in enumerate(state.candidates[:self.max_candidates], start=1)
            ],
        }
        return f"""
You are the central orchestrator of a fast multi-agent entity alignment system.
Choose one compact single-stage execution plan that balances accuracy and speed.

Important:
- Use listwise agents over candidate sets, not pairwise calls.
- There is no second stage. Decide all agents now.
- You may choose any non-empty combination of TextAgent, StructureAgent, VisualTextAgent, and ImageAgent.
- TextAgent sees only entity names. It can process top30.
- StructureAgent sees only structural evidence: neighbors, relations, graph context. It can process top30.
- VisualTextAgent sees only img.json textual image descriptions. It can process top30.
- ImageAgent sees only raw images and is expensive. Use it only when the case is hard, the top5 candidates remain difficult, and raw image evidence is likely necessary. ImageAgent should process at most top5.
- Prefer smaller plans for clear/easy cases, but combine agents when the evidence sources are complementary.
- Return JSON only. No markdown.

Schema:
{{
  "text_model": "{self.text_model_pool[0]}",
  "vision_model": "{self.vision_model_pool[0]}",
  "stages": [
    {{"agent": "TextAgent", "top_k": 30}},
    {{"agent": "VisualTextAgent", "top_k": 30}},
    {{"agent": "ImageAgent", "top_k": 5}}
  ],
  "reason": "short reason"
}}

Input:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    def _fallback_route(self, state: EntityState) -> Dict[str, Any]:
        return {
            "text_model": self.text_model_pool[0],
            "vision_model": self.vision_model_pool[0],
            "stages": [
                {"agent": "TextAgent", "top_k": min(self.default_text_top_k, len(state.candidates), self.max_candidates)},
                {"agent": "StructureAgent", "top_k": min(self.default_structure_top_k, len(state.candidates), self.max_candidates)},
            ],
            "reason": "fallback fast text+structure plan",
            "used_fallback": True,
        }

    def _sanitize_route(self, route: Dict[str, Any], state: EntityState) -> Dict[str, Any]:
        allowed = {"TextAgent", "StructureAgent", "VisualTextAgent", "ImageAgent"}
        max_by_agent = {
            "TextAgent": self.default_text_top_k,
            "StructureAgent": self.default_structure_top_k,
            "VisualTextAgent": self.default_visual_text_top_k,
            "ImageAgent": self.default_image_top_k,
        }
        stages = []
        for raw_stage in route.get("stages", []):
            if not isinstance(raw_stage, dict):
                continue
            agent = str(raw_stage.get("agent", "")).strip()
            if agent not in allowed:
                continue
            try:
                top_k = int(raw_stage.get("top_k", max_by_agent[agent]))
            except Exception:
                top_k = max_by_agent[agent]
            top_k = max(1, min(top_k, max_by_agent[agent], len(state.candidates), self.max_candidates))
            if not any(stage["agent"] == agent for stage in stages):
                stages.append({"agent": agent, "top_k": top_k})

        if not stages:
            stages = self._fallback_route(state)["stages"]

        has_main_visual_text = bool(state.main_entity.img_desc)
        has_candidate_visual_text = any(cand.img_desc for cand in state.candidates[:self.max_candidates])
        if not (has_main_visual_text and has_candidate_visual_text):
            stages = [stage for stage in stages if stage["agent"] != "VisualTextAgent"]

        has_main_raw_image = bool(state.main_entity.image_path)
        has_candidate_raw_image = any(cand.image_path for cand in state.candidates[: self.default_image_top_k])
        if not (has_main_raw_image and has_candidate_raw_image):
            stages = [stage for stage in stages if stage["agent"] != "ImageAgent"]

        if not stages:
            stages = self._fallback_route(state)["stages"]

        order = {"TextAgent": 0, "StructureAgent": 1, "VisualTextAgent": 2, "ImageAgent": 3}
        stages = sorted(stages, key=lambda stage: order.get(stage["agent"], 99))

        text_model = str(route.get("text_model", self.text_model_pool[0])).strip()
        vision_model = str(route.get("vision_model", self.vision_model_pool[0])).strip()
        if text_model not in self.text_model_pool:
            text_model = self.text_model_pool[0]
        if vision_model not in self.vision_model_pool:
            vision_model = self.vision_model_pool[0]

        return {
            "text_model": text_model,
            "vision_model": vision_model,
            "stages": stages,
            "reason": truncate_text(route.get("reason", "fast llm plan"), 160),
            "used_fallback": bool(route.get("used_fallback", False)),
        }


    def route(self, state: EntityState) -> Dict[str, Any]:
        prompt = self.build_prompt(state)
        st = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.orchestrator_model,
                messages=[
                    {"role": "system", "content": "You are a speed-aware central orchestrator for multi-agent entity alignment."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=450,
            )
            latency = time.time() - st
            usage = getattr(response, "usage", None)
            raw_text = (response.choices[0].message.content or "").strip()
            payload = extract_json_payload(raw_text)
            if not isinstance(payload, dict):
                payload = self._fallback_route(state)
            route = self._sanitize_route(payload, state)
            route["raw_response"] = raw_text
            route["orchestrator_token_total"] = int(getattr(usage, "total_tokens", 0) or 0)
            route["orchestrator_latency"] = latency
            return route
        except Exception as e:
            safe_print(f"[FastLLMOrchestrator] API error: {e}")
            route = self._sanitize_route(self._fallback_route(state), state)
            route["raw_response"] = ""
            route["orchestrator_token_total"] = 0
            route["orchestrator_latency"] = 0.0
            return route

    def run(self, state: EntityState) -> EntityState:
        route = self.route(state)
        state.orchestration_trace = route
        state.total_token_cost += int(route.get("orchestrator_token_total", 0))
        state.total_latency += float(route.get("orchestrator_latency", 0.0))

        agents = self.agents_factory(text_model=route["text_model"], vision_model=route["vision_model"])
        selected_agents = []
        execution_trace = []
        for stage in route["stages"]:
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


def build_candidate_feature_vector(state: EntityState, candidate: Candidate, outputs: List[AgentOutput]) -> List[float]:
    grouped: Dict[str, List[AgentOutput]] = {}
    for out in outputs:
        grouped.setdefault(out.agent_name, []).append(out)

    def avg_val(agent_name: str, attr: str) -> float:
        arr = [float(getattr(o, attr, 0.0)) for o in grouped.get(agent_name, [])]
        if not arr:
            return 0.0
        return sum(arr) / len(arr)

    def used(agent_name: str) -> float:
        return 1.0 if grouped.get(agent_name) else 0.0

    rank = 0
    for idx, cand in enumerate(state.candidates):
        if cand.ent_id == candidate.ent_id:
            rank = idx
            break

    feat = [
        float(candidate.base_score),
        min(rank / 30.0, 1.0),
        name_overlap_score(state.main_entity.name, candidate.name),
        compute_neighbor_overlap(state.main_entity.neighbors, candidate.neighbors),
        float(bool(candidate.img_desc)),
        float(bool(candidate.image_path)),
        min(len(candidate.neighbors), 50) / 50.0,
    ]
    for agent_name in ["TextAgent", "StructureAgent", "VisualTextAgent", "ImageAgent"]:
        feat.append(avg_val(agent_name, "score"))
        feat.append(avg_val(agent_name, "confidence"))
        feat.append(used(agent_name))
    return sanitize_feature_vector(feat)


class TrainableAggregator(nn.Module):
    BASE_NAME = "HHEA"
    AGENT_NAMES = ["TextAgent", "StructureAgent", "VisualTextAgent", "ImageAgent"]

    def __init__(self, input_dim: int, hidden_dim: int = 96, use_equal_weights: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.agent_count = len(self.AGENT_NAMES)
        self.source_names = [self.BASE_NAME] + self.AGENT_NAMES
        self.source_count = len(self.source_names)
        self.agent_block_start = input_dim - self.agent_count * 3
        if self.agent_block_start < 0:
            raise ValueError(f"input_dim={input_dim} is too small for agent-weight aggregation")
        self.source_logits = nn.Parameter(torch.ones(self.source_count))
        self.use_equal_weights = use_equal_weights  # 新增的参数，用于控制是否使用平等权重

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)

        base_score = feat[:, 0]
        agent_part = feat[:, self.agent_block_start:].view(-1, self.agent_count, 3)
        agent_score = agent_part[:, :, 0]
        agent_confidence = agent_part[:, :, 1]
        agent_used = agent_part[:, :, 2] > 0.5

        effective_score = agent_score * torch.clamp(agent_confidence, min=0.0, max=1.0)
        source_score = torch.cat([base_score.unsqueeze(-1), effective_score], dim=-1)
        base_used = torch.ones((feat.size(0), 1), dtype=torch.bool, device=feat.device)  #todo
        source_used = torch.cat([base_used, agent_used], dim=-1)
        logits = self.source_logits.unsqueeze(0).expand_as(source_score)
        #print(logits)

        # 如果启用平等权重
        if self.use_equal_weights:
            # 设置所有选中的来源的权重为相同，均为 1
            weights = torch.ones_like(source_score)  # 每个来源的权重都为 1
            #print(weights)
            weights = weights * source_used  # 仅对已选来源有效
        else:
            # 否则，继续使用 softmax 权重
            masked_logits = logits.masked_fill(~source_used, -1e9)
            weights = torch.softmax(masked_logits, dim=-1)

        return torch.sum(weights * source_score, dim=-1)

    def current_weights(self) -> Dict[str, float]:
        if self.use_equal_weights:
            weight = 1.0 / max(1, self.source_count)
            return {source_name: weight for source_name in self.source_names}
        with torch.no_grad():
            weights = torch.softmax(self.source_logits.detach().cpu(), dim=-1)
        return {
            source_name: float(weights[idx].item())
            for idx, source_name in enumerate(self.source_names)
        }

    def selected_weights(self, selected_agents: List[str]) -> Dict[str, float]:
        selected = {self.BASE_NAME, *set(selected_agents)}
        active_names = [source_name for source_name in self.source_names if source_name in selected]
        if not active_names:
            return {}
        if self.use_equal_weights:
            weight = 1.0 / max(1, len(active_names))
            return {source_name: weight for source_name in active_names}

        mask = torch.tensor(
            [source_name in selected for source_name in self.source_names],
            dtype=torch.bool,
            device=self.source_logits.device,
        )
        with torch.no_grad():
            masked_logits = self.source_logits.masked_fill(~mask, -1e9)
            weights = torch.softmax(masked_logits.detach().cpu(), dim=-1)
        return {
            source_name: float(weights[idx].item())
            for idx, source_name in enumerate(self.source_names)
            if source_name in selected
        }


class AggregationWrapper:
    def __init__(self, model: TrainableAggregator, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def score_candidate(self, state: EntityState, candidate: Candidate, outputs: List[AgentOutput]) -> float:
        feat = build_candidate_feature_vector(state, candidate, outputs)
        x = feature_tensor(feat, self.device)
        with torch.no_grad():
            s = self.model(x).item()
        return float(s)

    def rank(self, state: EntityState) -> List[Dict[str, Any]]:
        ranked = []
        for cand in state.candidates:
            outputs = state.agent_outputs.get(cand.ent_id, [])
            final_score = self.score_candidate(state, cand, outputs)
            ranked.append({
                "ent_id": cand.ent_id,
                "name": cand.name,
                "base_score": cand.base_score,
                "final_score": final_score,
                "agent_outputs": [
                    {
                        "agent_name": o.agent_name,
                        "score": o.score,
                        "confidence": o.confidence,
                        "rationale": o.rationale,
                    }
                    for o in outputs
                ],
            })
        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked

    def training_scores(self, state: EntityState) -> torch.Tensor:
        feats = []
        for cand in state.candidates:
            outputs = state.agent_outputs.get(cand.ent_id, [])
            feats.append(build_candidate_feature_vector(state, cand, outputs))
        return self.model(feature_matrix(feats, self.device))


def load_or_init_aggregator(input_dim: int, ckpt_path: Optional[str], device: str) -> TrainableAggregator:
    agg = TrainableAggregator(input_dim=input_dim, hidden_dim=64, use_equal_weights=False).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        load_state_dict_if_compatible(agg, ckpt_path, device, "aggregator")
    return agg


class MAORMasRouterReasoner:
    def __init__(self, orchestrator: Any, aggregator: Any):
        self.orchestrator = orchestrator
        self.aggregator = aggregator

    def rerank_with_state(self, main_entity: MainEntity, candidates: List[Candidate]) -> Tuple[List[Dict[str, Any]], EntityState]:
        state = EntityState(main_entity=main_entity, candidates=candidates)
        state = self.orchestrator.run(state)
        ranked = self.aggregator.rank(state)
        if hasattr(self.aggregator.model, "current_weights"):
            state.orchestration_trace["agent_weights"] = self.aggregator.model.current_weights()
        if hasattr(self.aggregator.model, "selected_weights"):
            state.orchestration_trace["selected_agent_weights"] = self.aggregator.model.selected_weights(state.selected_agents)
        return ranked, state

    def rerank(self, main_entity: MainEntity, candidates: List[Candidate]) -> List[Dict[str, Any]]:
        ranked, _ = self.rerank_with_state(main_entity, candidates)
        return ranked


def build_reasoner(
    client: openai.OpenAI,
    text_model_pool: List[str],
    vision_model_pool: List[str],
    agg_ckpt: Optional[str],
    device: str,
    *,
    vision_client: Optional[openai.OpenAI] = None,
) -> MAORMasRouterReasoner:
    agents_factory = build_listwise_agents_factory(client, vision_client)

    example_state = EntityState(main_entity=MainEntity(ent_id=0, name="x"), candidates=[])
    example_candidate = Candidate(ent_id=1, name="y", base_score=0.0)
    agg_input_dim = len(build_candidate_feature_vector(example_state, example_candidate, []))
    aggregator_model = load_or_init_aggregator(agg_input_dim, agg_ckpt, device)

    orchestrator = FastLLMOrchestrator(
        client=client,
        agents_factory=agents_factory,
        text_model_pool=text_model_pool,
        vision_model_pool=vision_model_pool,
        max_candidates=30,
        default_text_top_k=30,
        default_structure_top_k=30,
        default_visual_text_top_k=30,
        default_image_top_k=5,
    )
    aggregator = AggregationWrapper(aggregator_model, device=device)
    return MAORMasRouterReasoner(orchestrator, aggregator)
