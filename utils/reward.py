"""
Reward function for train_f1.py
"""

import re, json
import os
from typing import List, Dict, Any, Optional
from .prompts import ALLOWED_SHAPES
import numpy as np


# --- WandB logging -----------------------------------------------------------
def _wandb_log(data: Dict[str, Any]):
    try:
        import wandb
        if getattr(wandb, "run", None) is not None:
            wandb.log(data)
    except Exception:
        pass

def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if len(xs) else 0.0


# --- Completion logging -------------------------------------------------------
_COMPLETIONS_DIR: Optional[str] = None
_ROTATE_EVERY: int = 500
_CALLS: int = 0

def set_completions_dir_(base_dir: str, version: str = "v1", run: str = "run0", rotate_every: int = 500):
    """
    Set completions directory for a specific version/run.
    Files will be saved as:
      <base_dir>/<version>/<run>/part000.jsonl
    """
    global _COMPLETIONS_DIR, _ROTATE_EVERY, _CALLS
    _COMPLETIONS_DIR = os.path.join(base_dir, version, run)
    _ROTATE_EVERY = max(1, int(rotate_every))
    _CALLS = 0
    os.makedirs(_COMPLETIONS_DIR, exist_ok=True)

def _step_from(kwargs: Dict[str, Any]) -> Optional[int]:
    for k in ("global_step", "step", "iteration"):
        v = kwargs.get(k)
        if v is None:
            continue
        try:
            vi = int(v)
            if vi >= 0:
                return vi
        except Exception:
            pass
    try:
        import wandb
        if getattr(wandb, "run", None) is not None and hasattr(wandb.run, "step"):
            s = wandb.run.step
            if s is not None:
                return int(s)
    except Exception:
        pass
    return None

def _rotated_path(step: Optional[int]) -> Optional[str]:
    if not _COMPLETIONS_DIR:
        return None
    part = (step if step is not None else _CALLS) // _ROTATE_EVERY
    return os.path.join(_COMPLETIONS_DIR, f"part{part:03d}.jsonl")

def _log_completions(completions: List[str], completions_file: str = None, **kwargs):
    global _CALLS
    if not completions:
        return

    target = completions_file or _rotated_path(_step_from(kwargs))
    if not target:
        return

    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"BATCH: {len(completions)} completions\n")
            s = _step_from(kwargs)
            if s is not None:
                f.write(f"STEP: {s}\n")
            f.write("=" * 80 + "\n")
            for i, c in enumerate(completions, 1):
                f.write(f"\n--- COMPLETION {i} ---\n")
                f.write(c)
                f.write("\n")
            f.write("\n")
    except Exception as e:
        print(f"Warning: Could not save completions to {target}: {e}")
    finally:
        _CALLS += 1


# --- Helper functions ---------------------------------------------------------
TAG_RE = re.compile(r"^<answer>\s*(\[.*\])\s*</answer>\s*$", re.DOTALL)

def _extract_json_block(text: str) -> Optional[str]:
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        lines = text.split('\n')
        if len(lines) >= 3:
            text = '\n'.join(lines[1:-1])
    elif text.startswith('```json') and text.endswith('```'):
        lines = text.split('\n')
        if len(lines) >= 3:
            text = '\n'.join(lines[1:-1])
    m = TAG_RE.match(text.strip())
    return m.group(1) if m else None

def _is_int(v) -> bool:
    try:
        return float(v).is_integer()
    except Exception:
        return isinstance(v, int)

def _validate_item(d: Dict[str, Any]) -> float:
    """ Schema validator with binary validation """
    if not isinstance(d, dict):
        return 0.2

    if "shape" not in d or not isinstance(d["shape"], str):
        return 0.2
    if d["shape"] not in ALLOWED_SHAPES:
        return 0.2

    c = d.get("color", "")
    if not isinstance(c, str) or not c.startswith("#") or len(c) != 7:
        return 0.2

    return 1.0

def _safe_load_json(arr_str: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse JSON str to check :
    - Top-level object is list
    - Each element is dict
    """
    try:
        obj = json.loads(arr_str)
        if isinstance(obj, list) and len(obj) >= 1 and all(isinstance(x, dict) for x in obj):
            return obj
        return None
    except Exception:
        return None

def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return r, g, b
    except ValueError:
        raise ValueError(f"Invalid hex color: {hex_color}")

def _rgb_to_xyz(r: int, g: int, b: int) -> tuple:
    """Convert RGB to XYZ color space"""
    # Normalize RGB to 0-1
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # Apply gamma correction
    def gamma_correct(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    r, g, b = gamma_correct(r), gamma_correct(g), gamma_correct(b)

    # Convert to XYZ using sRGB matrix
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    return x, y, z

def _xyz_to_lab(x: float, y: float, z: float) -> tuple:
    """Convert XYZ to LAB color space"""
    # D65 illuminant
    xn, yn, zn = 0.95047, 1.0, 1.08883

    x, y, z = x / xn, y / yn, z / zn

    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t + 16/116)

    fx, fy, fz = f(x), f(y), f(z)

    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return l, a, b

def _rgb_to_lab(r: int, g: int, b: int) -> tuple:
    """Convert RGB directly to LAB"""
    x, y, z = _rgb_to_xyz(r, g, b)
    return _xyz_to_lab(x, y, z)

def _color_score(a: str, b: str) -> float:
    """
    Perceptual color distance using Delta E in CIELAB space.
    1.0 if colors are identical.
    0.0 if Delta E >= 100 (very different colors).
    """
    # Validate color strings
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.2  # Invalid color

    try:
        r_a, g_a, b_a = _hex_to_rgb(a)
        r_b, g_b, b_b = _hex_to_rgb(b)

        lab_a = _rgb_to_lab(r_a, g_a, b_a)
        lab_b = _rgb_to_lab(r_b, g_b, b_b)

        # Euclidean distance in LAB space
        delta_e = ((lab_a[0] - lab_b[0]) ** 2 +
                   (lab_a[1] - lab_b[1]) ** 2 +
                   (lab_a[2] - lab_b[2]) ** 2) ** 0.5

        base_score = max(0.0, 1.0 - (delta_e / 100.0))
        return base_score
    except Exception as e:
        print(f"🚨Fallback to original RGB distance if conversion fails: {e}")
        try:
            r_a, g_a, b_a = _hex_to_rgb(a)
            r_b, g_b, b_b = _hex_to_rgb(b)
            d = (abs(r_a - r_b) + abs(g_a - g_b) + abs(b_a - b_b)) / (3 * 255)
            base_score = 1.0 - d
            return base_score
        except Exception:
            return 0.2  # If fallback also fails, return 0.2

def _shape_factor(pred_shape: str, ref_shape: str) -> float:
    if pred_shape == ref_shape:
        return 1.0
    return 0.2


# --- Similarity matrix builder (shared) ---------------------------------------
def _build_similarity_matrix(pred_list: List[Dict[str, Any]], ref_list: List[Dict[str, Any]]) -> tuple[np.ndarray, List[float], List[float]]:
    n, m = len(pred_list), len(ref_list)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=float), [], []

    S = np.zeros((n, m), dtype=float)
    c_scores, sh_scores = [], []
    
    for i, p in enumerate(pred_list):
        for j, r in enumerate(ref_list):
            p_color = p.get("color", "")
            r_color = r.get("color", "")
            p_shape = p.get("shape", "")
            r_shape = r.get("shape", "")
            c_score = _color_score(p_color, r_color) if p_color and r_color else 0.2
            sh_score = _shape_factor(p_shape, r_shape) if p_shape and r_shape else 0.2
            
            c_scores.append(c_score)
            sh_scores.append(sh_score)
            S[i, j] = float(max(0.0, min(1.0, 0.7 * c_score + 0.3 * sh_score)))
    return S, c_scores, sh_scores


# --- Set-F1 scorer ------------------------------------------------------------
def set_f1_score(S: np.ndarray, tau: float = 0.6) -> float:
    n, m = S.shape
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0

    used_p = np.zeros(n, dtype=bool)
    used_r = np.zeros(m, dtype=bool)
    tps = 0

    # Collect candidate pairs >= tau
    pairs = [(i, j, S[i, j]) for i in range(n) for j in range(m) if S[i, j] >= tau]
    # High → low similarity
    pairs.sort(key=lambda x: x[2], reverse=True)

    for i, j, _ in pairs:
        if not used_p[i] and not used_r[j]:
            used_p[i] = True
            used_r[j] = True
            tps += 1

    precision = tps / max(1, n)  # matched / total preds
    recall = tps / max(1, m)     # matched / total refs
    
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --- Weighted helper -----------------------------------------------------------
def weighted_(reward_callable, weight: float):
    """
    Helper function for weighted sum.
    """
    name = getattr(reward_callable, "name", reward_callable.__class__.__name__.lower())

    def f(completions: List[str], **kw) -> List[float]:
        raw = reward_callable(completions, **kw)
        out = [weight * x for x in raw]
        _wandb_log({f"reward_weighted/{name}": _mean(out)})
        return out
    return f


# --- Reward classes ------------------------------------------------------------
class FormatReward_:
    """
    Format reward (0-1):
      - tags present (0.3)
      - JSON parses (0.3)
      - schema valid (0.4)
    """
    def __init__(self, w_tags=0.3, w_json=0.3, w_schema=0.4):
        self.w_tags = w_tags
        self.w_json = w_json
        self.w_schema = w_schema
        self.name = "format"

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        tags_flags, json_flags, schema_flags = [], [], []

        # Log completions to file
        _log_completions(completions, **kwargs)

        for content in completions:
            tags_ok = 1.0 if TAG_RE.match(content.strip()) else 0.2
            arr = _extract_json_block(content) if tags_ok else None
            json_ok = 1.0 if (arr and _safe_load_json(arr) is not None) else 0.2
            schema_ok = 0.2
            if json_ok:
                items = _safe_load_json(arr)
                if items:
                    # Average validation scores for all items
                    scores = [_validate_item(it) for it in items]
                    schema_ok = sum(scores) / len(scores) if scores else 0.2

            score = self.w_tags * tags_ok + self.w_json * json_ok + self.w_schema * schema_ok
            rewards.append(score)
            tags_flags.append(tags_ok); json_flags.append(json_ok); schema_flags.append(schema_ok)

        # wandb logging (means)
        _wandb_log({
            "fmt_reward/mean": _mean(rewards),
            "fmt_reward/tags": _mean(tags_flags),
            "fmt_reward/json": _mean(json_flags),
            "fmt_reward/schema": _mean(schema_flags),
        })
        return rewards


class AccuracyReward_:
    """
    Accuracy reward (0-1) using F1 
    """
    def __init__(self, reference_key: str = "solution", tau: float = 0.6):
        self.reference_key = reference_key
        self.name = "accuracy"
        self.tau = tau  # similarity threshold for counting a match

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        refs = kwargs.get(self.reference_key, None)
        scores: List[float] = []
        all_c_scores: List[float] = []
        all_sh_scores: List[float] = []

        for i, content in enumerate(completions):
            ref_list = refs[i] if refs is not None else []
            arr = _extract_json_block(content)
            if not arr:
                scores.append(0.2); continue
            pred_list = _safe_load_json(arr)
            if not pred_list:
                scores.append(0.2); continue

            # Check if items have valid structure
            item_scores = [_validate_item(x) for x in pred_list]
            if not any(score > 0 for score in item_scores):
                scores.append(0.2); continue

            # Build similarity and score with Set-F1
            S, c_scores, sh_scores = _build_similarity_matrix(pred_list, ref_list)
            s = set_f1_score(S, tau=self.tau)
            scores.append(s)
            
            # Collect scores for logging
            all_c_scores.extend(c_scores)
            all_sh_scores.extend(sh_scores)

        _wandb_log({
            "acc_reward/mean": _mean(scores),
            "acc_reward/c_score_mean": _mean(all_c_scores),
            "acc_reward/sh_score_mean": _mean(all_sh_scores),
            "acc_reward/tau": self.tau
        })
        return scores
