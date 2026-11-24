import base64
import json
import math
import os
import re
from typing import Any, List, Sequence, Tuple

import torch
import torch.nn as nn


LOG_PRINT = False


def safe_print(*args, **kwargs) -> None:
    if LOG_PRINT:
        print(*args, **kwargs)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def extract_float(text: str, key: str, default: float = 0.5) -> float:
    pattern = rf"\[\s*{re.escape(key)}\s*:\s*([0-9]*\.?[0-9]+)\s*\]"
    match = re.search(pattern, text, re.I)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return default
    return default


def extract_text(text: str, key: str, default: str = "") -> str:
    pattern = rf"\[\s*{re.escape(key)}\s*:\s*(.*?)\s*\]"
    match = re.search(pattern, text, re.I | re.S)
    if match:
        return match.group(1).strip()
    return default


def extract_json_payload(text: str) -> Any:
    if not text:
        return None

    candidates = [text.strip()]
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    object_match = re.search(r"\{[\s\S]*\}", text)
    if object_match:
        candidates.append(object_match.group(0).strip())

    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        candidates.append(array_match.group(0).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def truncate_text(text: Any, limit: int = 160) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def convert_neighbor_to_str(nei: Any) -> str:
    if isinstance(nei, (list, tuple)):
        return "(" + ", ".join(str(x) for x in nei) + ")"
    return str(nei)


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def safe_name_tokens(name: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", (name or "").lower())
    return toks


def name_overlap_score(name1: str, name2: str) -> float:
    return jaccard(safe_name_tokens(name1), safe_name_tokens(name2))


def compute_neighbor_overlap(main_neighbors: List[Any], cand_neighbors: List[Any]) -> float:
    if not main_neighbors or not cand_neighbors:
        return 0.0
    main_set = {convert_neighbor_to_str(x) for x in main_neighbors}
    cand_set = {convert_neighbor_to_str(x) for x in cand_neighbors}
    inter = len(main_set & cand_set)
    union = len(main_set | cand_set)
    if union == 0:
        return 0.0
    return clamp(inter / union)


def save_json_atomic(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fw:
        json.dump(obj, fw, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def transform_time(seconds: int) -> tuple[int, int, int]:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds


def resolve_torch_device(requested_device: str) -> str:
    req = (requested_device or "auto").strip().lower()
    if req == "auto":
        req = "cuda" if torch.cuda.is_available() else "cpu"

    if not req.startswith("cuda"):
        return "cpu"

    if not torch.cuda.is_available():
        print("Warning: CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"

    try:
        probe_device = torch.device(req)
        x = torch.randn(1, 36, device=probe_device, dtype=torch.float32)
        layer = nn.Linear(36, 64).to(probe_device)
        y = layer(x)
        _ = float(y.sum().item())
        return req
    except Exception as e:
        print(f"Warning: CUDA sanity check failed on device '{req}' ({e}). Falling back to CPU.")
        return "cpu"


def sanitize_feature_vector(values: List[float]) -> List[float]:
    clean = []
    for value in values:
        try:
            v = float(value)
        except Exception:
            v = 0.0
        if math.isnan(v) or math.isinf(v):
            v = 0.0
        clean.append(v)
    return clean


def feature_tensor(values: List[float], device: str) -> torch.Tensor:
    return torch.tensor(sanitize_feature_vector(values), dtype=torch.float32, device=device).unsqueeze(0)


def feature_matrix(values: List[List[float]], device: str) -> torch.Tensor:
    rows = [sanitize_feature_vector(row) for row in values]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def load_state_dict_if_compatible(module: nn.Module, ckpt_path: str, device: str, module_name: str) -> None:
    loaded = torch.load(ckpt_path, map_location=device)
    state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
    if not isinstance(state_dict, dict):
        print(f"Warning: invalid checkpoint format for {module_name}: {ckpt_path}. Reinitializing.")
        return

    current = module.state_dict()
    mismatched = []
    for key, tensor in state_dict.items():
        if key not in current:
            mismatched.append(f"{key}:missing")
            continue
        if tuple(tensor.shape) != tuple(current[key].shape):
            mismatched.append(f"{key}:{tuple(tensor.shape)}!={tuple(current[key].shape)}")

    if mismatched:
        preview = ", ".join(mismatched[:3])
        print(
            f"Warning: checkpoint '{ckpt_path}' is incompatible with current {module_name} architecture "
            f"({preview}). Reinitializing {module_name}."
        )
        return

    module.load_state_dict(state_dict)


def evaluate_alignment(
    ranks: Sequence[int], hit_k: Sequence[int] = (1, 5, 10)
) -> Tuple[List[float], float]:
    """Compute Hits@K and MRR from zero-based ranks."""
    count = len(ranks)
    hits = [round(sum(rank < k for rank in ranks) / count, 4) for k in hit_k]
    reciprocal_rank_sum = 0.0
    for rank in ranks:
        reciprocal_rank_sum += 1 / (rank + 1)
    return hits, reciprocal_rank_sum / count
