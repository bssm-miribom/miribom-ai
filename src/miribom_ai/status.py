"""모델 상태 요약 (서버 /api/admin/models 용). 무거운 라이브러리를 불러오지 않고 파일만 읽음"""
import csv, json
from datetime import datetime
from pathlib import Path

from miribom_ai.paths import ROOT

HORIZONS = (1, 2, 3)
WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def _mtime(path: Path):
    return datetime.fromtimestamp(path.stat().st_mtime).date().isoformat() if path.exists() else None


def _scale_metrics(path: Path):
    """results/scale_metrics.csv 의 D+1 전체 구간 WAPE (운영 모델·기준 모델)"""
    if not path.exists():
        return None
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["예측"] != "D+1" or r["구간"] != "전체":
                continue
            m = r["모델"]
            key = ("d1_wape" if "운영" in m else "baseline_prev_day" if "전날" in m
                   else "baseline_weekday" if "요일" in m else None)
            if key and key not in out:
                out[key] = round(float(r["WAPE"]), 3)
                if key == "d1_wape":
                    out["d1_coverage"] = round(float(r["범위적중률"]), 3)
    return out or None


def scale_status(demo=False):
    d = ROOT / "models" / ("demo" if demo else "")
    ready = all((d / f"scale_h{h}_meta.json").exists() for h in HORIZONS)
    meta = json.load(open(d / "scale_h1_meta.json")) if ready else {}
    return {"ready": ready, "demo": demo, "trained_at": _mtime(d / "scale_h1_q50.txt"),
            "features": meta.get("features", []), "surge_elasticity": meta.get("surge_elasticity"),
            "metrics": _scale_metrics(ROOT / "results" / ("scale_metrics_demo.csv" if demo else "scale_metrics.csv"))}


def intent_status():
    d = ROOT / "models" / "intent"
    th = json.load(open(d / "threshold.json")) if (d / "threshold.json").exists() else {}
    has_weights = any((d / w).exists() for w in WEIGHTS)
    ready = has_weights and bool(th)
    try:
        import torch, transformers  # noqa: F401
        libs = True
    except ImportError:
        libs = False
    reason = None if ready and libs else ("모델 가중치 파일 없음 (models/intent/model.safetensors)" if not has_weights
                                          else "threshold.json 없음" if not th else "torch·transformers 미설치")
    return {"ready": ready and libs, "name": th.get("model"), "threshold": th.get("threshold"),
            "trained_at": th.get("trained_at") or _mtime(d / "threshold.json"),
            "metrics": {k: th.get(k) for k in ("f1", "keyword_f1", "n_train", "n_test") if th.get(k) is not None} or None,
            "labelers": th.get("labelers"), "reason": reason}


def intent_counts_status():
    p = ROOT / "data" / "intent_counts.csv"
    if not p.exists():
        return {"ready": False, "last_date": None}
    with open(p, encoding="utf-8-sig") as f:
        dates = [r["date"] for r in csv.DictReader(f) if r.get("date")]
    return {"ready": bool(dates), "last_date": max(dates) if dates else None}


def model_status(demo=False):
    return {"scale": scale_status(demo), "intent": intent_status(), "intent_counts": intent_counts_status(),
            "surge": {"ready": True, "method": "통계 (지난 1년 상위 1% 2일 연속 또는 1년 최댓값 초과)"}}
