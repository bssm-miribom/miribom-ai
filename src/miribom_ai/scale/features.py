"""입력 표에서 학습용 특징과 정답을 만듦. 판정일 07시 전에 알 수 있는 값만 씀"""
import numpy as np, pandas as pd
from miribom_ai.scale.config import HORIZONS, VIS_LAG, BASE_DAYS, INTEREST

def daytype(dates, holiday):
    return np.where((dates.dt.dayofweek >= 5) | (holiday == 1), "rest", "work")

def build(df):
    df = df.sort_values(["area", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    if "intent_posts" not in df: df["intent_posts"] = np.nan
    df["dtype"] = daytype(df["date"], df["holiday"])
    out = []
    for area, g in df.groupby("area"):
        g = g.set_index("date").asfreq("D")
        g["area"] = area
        # 관심도: 급증 배수 (최근 8주 같은 날 유형 중앙값 대비), 증가 속도
        si = g["search_index"]
        base_si = pd.Series(np.nan, index=g.index)
        for t in ("rest", "work"):
            m = g["dtype"] == t
            base_si[m] = si.where(m).shift(1).rolling(56, min_periods=10).median()[m]
        g["ratio"] = si / base_si
        g["ratio_max3"] = g["ratio"].rolling(3).max()
        g["ratio_growth3"] = g["ratio"] / g["ratio"].shift(3)
        g["intent_growth3"] = (g["intent_posts"] + 1) / (g["intent_posts"].shift(3) + 1)
        g["prev_visitors"] = g["visitors"]                  # 기준 모델(전날 값)용. 모델 입력에는 안 씀
        # 평소 방문객: 28~84일 전 같은 날 유형 중앙값 (최근 실측은 아직 공개 안 됐다고 봄)
        vis = g["visitors"]
        for t in ("rest", "work"):
            g[f"base_{t}"] = vis.where(g["dtype"] == t).shift(VIS_LAG).rolling(BASE_DAYS - VIS_LAG, min_periods=8).median()
        for h in HORIZONS:
            off = h + 1
            tgt_dt = g["dtype"].shift(-off)
            g[f"base_h{h}"] = np.where(tgt_dt == "rest", g["base_rest"], g["base_work"])
            g[f"y_vis_h{h}"] = vis.shift(-off)
            td = g.index + pd.Timedelta(days=off)
            g[f"dow_h{h}"] = td.dayofweek
            g[f"hol_h{h}"] = g["holiday"].shift(-off)
            g[f"rest_h{h}"] = (tgt_dt == "rest").astype(float)
            # 연휴 몇째 날, 다음 휴일까지 남은 날 (휴일 달력은 미리 알 수 있는 정보라 누수 아님)
            rest = (g["dtype"] == "rest").astype(int)
            run = rest.groupby((rest != rest.shift()).cumsum()).cumsum() * rest
            g[f"run_h{h}"] = run.shift(-off)
            nxt = pd.Series(np.where(rest == 1, g.index, pd.NaT), index=g.index).bfill()
            g[f"to_rest_h{h}"] = ((nxt - g.index).dt.days).shift(-off)
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)

def features(h, cols, with_interest=True):
    # 지역 이름은 넣지 않음. 처음 보는 장소에도 쓰려고 평소 방문객 규모(log_base)로만 구분함
    f = [f"dow_h{h}", f"hol_h{h}", f"rest_h{h}", f"run_h{h}", f"to_rest_h{h}", "log_base"]
    f += [c for c in (f"rain_h{h}", f"tmax_h{h}") if c in cols]
    if with_interest:
        f += [c for c in INTEREST if c in cols]
    return f

