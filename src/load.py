import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    ent_id: int
    name: str
    base_score: float
    img_desc: str = ""
    neighbors: List[Any] = field(default_factory=list)
    image_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MainEntity:
    ent_id: int
    name: str
    img_desc: str = ""
    neighbors: List[Any] = field(default_factory=list)
    image_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


class NeighborGenerator:
    """Load the candidate exports consumed by the reranking pipeline."""

    def __init__(self, data: str, data_dir: str, cand_file: str = "cand"):
        directory = Path(data_dir) / data / "candidates"
        self.candidates = {
            int(ent_id): row for ent_id, row in _read_json(directory / cand_file).items()
        }

        neighbors = _read_json(directory / "neighbors")
        self.neighbors = {int(ent_id): rows for ent_id, rows in neighbors.items()}
        names = _read_json(directory / "name_dict")
        self.entity_names = names["ent"]
        self.relation_names = names["rel"]
        self.time_names = names.get("time", {})
        self.image_scores = _read_json(directory / "pair_mmea_sims")

    def get_entities(self) -> List[int]:
        return sorted(self.candidates)

    def get_ref_ent(self, ent_id: int) -> int:
        return int(self.candidates[ent_id]["ref"])

    def get_base_rank(self, ent_id: int) -> int:
        return int(self.candidates[ent_id]["ground_rank"])

    def get_neighbors(self, ent_id: int, neigh_num: int = 0) -> Dict[str, Any]:
        triples = self.neighbors.get(ent_id, [])
        if neigh_num > 0:
            triples = triples[:neigh_num]
        neighbors = []
        for head, relation, tail, *times in triples:
            neighbors.append((
                self.entity_names[str(head)],
                self.relation_names[str(relation)],
                self.entity_names[str(tail)],
                *(self.time_names[str(time_id)] for time_id in times),
            ))
        return {
            "ent_id": ent_id,
            "name": self.entity_names[str(ent_id)],
            "neighbors": neighbors,
        }

    def get_candidates(self, ent_id: int, neigh_num: int = 0) -> List[Dict[str, Any]]:
        row = self.candidates[ent_id]
        candidates = []
        for raw_id, score in zip(row["candidates"], row["cand_sims"]):
            candidate_id = int(raw_id)
            candidate = self.get_neighbors(candidate_id, neigh_num)
            left, right = sorted((ent_id, candidate_id))
            candidate["hhea_sim"] = round(float(score), 3)
            candidate["mmea_sim"] = self.image_scores.get(f"{left}-{right}", 0.0)
            candidates.append(candidate)
        return candidates


def load_image_descriptions(path: Optional[str]) -> Dict[int, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        candidates = [
            os.path.basename(path),
            str(Path(__file__).resolve().parent.parent / os.path.basename(path)),
        ]
        path = next((candidate for candidate in candidates if os.path.exists(candidate)), "")
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as fr:
        raw = json.load(fr)

    desc: Dict[int, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                ent_id = int(key)
            except Exception:
                continue
            if isinstance(value, str):
                desc[ent_id] = value
            elif isinstance(value, dict):
                text = value.get("desc", value.get("description", value.get("caption", "")))
                if text:
                    desc[ent_id] = str(text)
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            raw_id = item.get("ent_id", item.get("id", item.get("entity_id")))
            try:
                ent_id = int(raw_id)
            except Exception:
                continue
            text = item.get("desc", item.get("description", item.get("caption", "")))
            if text:
                desc[ent_id] = str(text)
    return desc


def get_candidate_image_path(dataset_name: str, role: str, ent_id: int, image_root: Optional[str] = None) -> Optional[str]:
    if image_root is None:
        return None
    path = os.path.join(image_root, f"{ent_id}.png")
    return path if os.path.exists(path) else None


def convert_main_entity(raw: Dict[str, Any], dataset_name: str, image_root: Optional[str], image_desc: Optional[Dict[int, str]] = None) -> MainEntity:
    image_path = raw.get("image_path")
    if image_path is None:
        image_path = get_candidate_image_path(dataset_name, "main", raw["ent_id"], image_root)
    ent_id = raw["ent_id"]
    img_desc = raw.get("img_desc", raw.get("image_desc", ""))
    if not img_desc and image_desc:
        img_desc = image_desc.get(int(ent_id), "")
    return MainEntity(
        ent_id=ent_id,
        name=raw.get("name", str(ent_id)),
        img_desc=img_desc,
        neighbors=raw.get("neighbors", []),
        image_path=image_path,
        extra=raw,
    )


def convert_candidates(raw_candidates: List[Dict[str, Any]], main_entity: MainEntity, dataset_name: str, image_root: Optional[str], image_desc: Optional[Dict[int, str]] = None) -> List[Candidate]:
    candidates: List[Candidate] = []
    for c in raw_candidates:
        image_path = c.get("image_path")
        if image_path is None:
            image_path = get_candidate_image_path(dataset_name, "candidate", c["ent_id"], image_root)
        ent_id = c["ent_id"]
        img_desc = c.get("img_desc", c.get("image_desc", ""))
        if not img_desc and image_desc:
            img_desc = image_desc.get(int(ent_id), "")
        cand = Candidate(
            ent_id=ent_id,
            name=c.get("name", str(ent_id)),
            base_score=float(c.get("hhea_sim", c.get("mmea_sim", 0.0))),
            img_desc=img_desc,
            neighbors=c.get("neighbors", []),
            image_path=image_path,
            extra=dict(c),
        )
        candidates.append(cand)
    candidates.sort(key=lambda x: x.base_score, reverse=True)
    return candidates


def build_entity_example(
    ng: NeighborGenerator,
    ent_id: int,
    dataset_name: str,
    image_root: Optional[str],
    image_desc: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    raw_main = ng.get_neighbors(ent_id, 5)
    raw_cands = ng.get_candidates(ent_id, 5)
    ref_ent = ng.get_ref_ent(ent_id)
    base_rank = ng.get_base_rank(ent_id)
    main_entity = convert_main_entity(raw_main, dataset_name, image_root, image_desc)
    candidates = convert_candidates(raw_cands, main_entity, dataset_name, image_root, image_desc)
    return {
        "ent_id": int(ent_id),
        "ref_ent": int(ref_ent),
        "base_rank": int(base_rank),
        "main_entity": main_entity,
        "candidates": candidates,
    }
