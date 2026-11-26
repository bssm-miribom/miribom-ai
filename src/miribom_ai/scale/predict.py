"""학습된 규모 예측 모델로 D+1~3 방문객 범위를 냄. 특징은 학습 때와 같은 build()로 만듦"""
import json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from miribom_ai.scale.config import HORIZONS, QUANTILES, SURGE_ELASTICITY
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
            cal = None
            if meta.get("cal_features") and (MODEL_DIR / f"scale_h{h}_cal_q90.txt").exists():
                cal = (meta["cal_features"], lgb.Booster(model_file=str(MODEL_DIR / f"scale_h{h}_cal_q90.txt")))
            _cache[h] = (meta["features"], boosters, cal, float((meta.get("interval_cal") or {}).get("c", 0.0)))
    return _cache

def surge_mult(ratio, elasticity=SURGE_ELASTICITY):
    """검색 급증 배수 → 방문 배수. 평소보다 낮은 날은 평소 모델에 맡기고 1로 둠"""
    r = np.nan_to_num(np.asarray(ratio, dtype=float), nan=1.0)
    return np.clip(r, 1.0, None) ** elasticity

def _surge_range(feat):
    """하위·중앙·상위에 곱할 급증 배수 (D-1 기준)
    하위: 최근 3일 중 가장 낮은 배수 (관심이 식는 경우)
    중앙: D-1 배수가 이어지는 경우
    상위: 최근 7일 중 가장 높았던 관심 수준이 다시 오는 경우 (증가 속도로 늘려 잡으면 첫날 급증에서 수천 배가 됨)"""
    r = feat["ratio"].dropna()
    if r.empty:
        return 1.0, 1.0, 1.0
    last = float(r.iloc[-1])
    lo = float(r.iloc[-3:].min())
    up = max(last, float(r.iloc[-7:].max()))
    return tuple(float(surge_mult(x)) for x in (lo, last, up))

def uses_weather():
    """지금 모델이 날씨(rain_h*, tmax_h*)를 입력으로 쓰는지"""
    return models_ready() and any(f.startswith("rain_h") for feats, *_ in load_models().values() for f in feats)

def models_ready():
    return all((MODEL_DIR / f"scale_h{h}_meta.json").exists() for h in HORIZONS)

def predict_place(search, base_date, holidays, base_rest, base_work, intent=None, weather=None):
    """search: base_date(D-1)까지의 검색 지수 시계열. 평소 방문객은 places.csv 값을 씀
    weather: pipeline.weather.forecast() 결과 {rain_h1, tmax_h1, ...}. 날씨를 입력으로 학습한 모델일 때만 쓰임"""
    base_date = pd.Timestamp(base_date)
    idx = pd.date_range(search.index.min(), base_date + pd.Timedelta(days=max(HORIZONS) + 1))
    df = pd.DataFrame({"date": idx, "area": "x", "visitors": np.nan,
                       "search_index": search.reindex(idx).to_numpy(),
                       "holiday": [int(d in holidays) for d in idx]})
    if intent is not None:
        df["intent_posts"] = intent.reindex(idx).to_numpy()
    feat = build(df).set_index("date")
    row = feat.loc[base_date].copy()
    for k, v in (weather or {}).items():
        row[k] = v
    out = {}
    s_lo, s_mid, s_hi = _surge_range(feat.loc[:base_date])
    for h, (feats, boosters, cal, c) in load_models().items():
        # 예전(10-31 오전) 모델: 관심도 없는 평소 모델만 있으면 곱해서 씀. 지금 모델은 학습값 그대로 쓰고 곱셈은 최악 시나리오에만
        multiply = cal is None and not any(f in feats for f in ("ratio", "ratio_max3"))
        base = base_rest if row[f"rest_h{h}"] == 1 else base_work
        row["log_base"] = np.log(base)
        x = pd.DataFrame([[row.get(f, np.nan) for f in feats]], columns=feats).astype(float)
        p = {q: float(base * np.exp(b.predict(x)[0])) for q, b in boosters.items()}
        lo, mid, hi = min(p[0.1], p[0.5]), p[0.5], max(p[0.9], p[0.5])
        # 범위 보정(scale/calibrate.py): 검증 기간 잔차로 구한 계수만큼 하위10·상위90을 넓힘. 중앙값은 그대로
        lo, hi = lo * np.exp(-c), hi * np.exp(c)
        if multiply:
            lo, mid, hi = lo * s_lo, mid * s_mid, hi * s_hi
        out[h] = {"날짜": str((base_date + pd.Timedelta(days=h + 1)).date()),
                  "하위10": round(lo), "중앙값": round(mid), "상위90": round(hi)}
        # 최악 시나리오: 평소 수준 모델 상위90 × 최근 7일 최대 검색 배수 ** SURGE_ELASTICITY (검색이 N배면 방문도 N배인 가정)
        # 실측 모델(위 하위10~상위90)과 별도로, 이 값의 밀집도가 5명/㎡를 넘으면 2단계 (alert/judge.py WORST_STAGE2)
        if cal is not None:
            cf, cb = cal
            xc = pd.DataFrame([[row.get(f, np.nan) for f in cf]], columns=cf).astype(float)
            out[h]["최악"] = round(max(hi, base * float(np.exp(cb.predict(xc)[0])) * s_hi))
    return out, row
