"""학습된 규모 예측 모델로 D+1~3 방문객 범위를 냄. 특징은 학습 때와 같은 build()로 만듦"""
import json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from miribom_ai.scale.config import HORIZONS, QUANTILES
from miribom_ai.scale.features import build

from miribom_ai.paths import ROOT
MODEL_DIR = ROOT / "models"      # 가짜 데이터로 시험할 때는 use_demo()로 models/demo/를 씀
_cache = {}

def use_demo(flag=True):
    """True면 가짜 데이터 모델(models/demo/), False면 실제 모델(models/)"""
    global MODEL_DIR
    new = ROOT / "models" / ("demo" if flag else "")
    if new != MODEL_DIR:
        MODEL_DIR = new; _cache.clear()

def load_models():
    if not _cache:
        for h in HORIZONS:
            meta = json.load(open(MODEL_DIR / f"scale_h{h}_meta.json"))
            boosters = {q: lgb.Booster(model_file=str(MODEL_DIR / f"scale_h{h}_q{int(q*100)}.txt")) for q in QUANTILES}
            _cache[h] = (meta["features"], boosters)
    return _cache

def models_ready():
    return all((MODEL_DIR / f"scale_h{h}_meta.json").exists() for h in HORIZONS)

def predict_place(search, base_date, holidays, base_rest, base_work, weather=None, intent=None):
    """search: base_date(D-1)까지의 검색 지수 시계열. 평소 방문객은 places.csv 값을 씀"""
    base_date = pd.Timestamp(base_date)
    idx = pd.date_range(search.index.min(), base_date + pd.Timedelta(days=max(HORIZONS) + 1))
    df = pd.DataFrame({"date": idx, "area": "x", "visitors": np.nan,
                       "search_index": search.reindex(idx).to_numpy(),
                       "holiday": [int(d in holidays) for d in idx]})
    if intent is not None:
        df["intent_posts"] = intent.reindex(idx).to_numpy()
    row = build(df).set_index("date").loc[base_date].copy()
    for k, v in (weather or {}).items():
        row[k] = v
    out = {}
    for h, (feats, boosters) in load_models().items():
        base = base_rest if row[f"rest_h{h}"] == 1 else base_work
        row["log_base"] = np.log(base)
        x = pd.DataFrame([[row.get(f, np.nan) for f in feats]], columns=feats).astype(float)
        p = {q: float(base * np.exp(b.predict(x)[0])) for q, b in boosters.items()}
        lo, mid, hi = min(p[0.1], p[0.5]), p[0.5], max(p[0.9], p[0.5])
        out[h] = {"날짜": str((base_date + pd.Timedelta(days=h + 1)).date()),
                  "하위10": round(lo), "중앙값": round(mid), "상위90": round(hi)}
    return out, row
