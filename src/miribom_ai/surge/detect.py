"""급증 배수 계산과 1단계 판정"""
import numpy as np, pandas as pd
from miribom_ai.surge.config import BASELINE_WEEKS, TOP_PCT, CONSEC_DAYS, HISTORY_DAYS

# ===== 급증 배수 =====
def surge_ratio(s):
    """급증 배수 = 그날 지수 / 최근 8주 같은 요일 유형(평일·주말) 중앙값. 그날은 평소 값에서 뺌"""
    is_weekend = s.index.dayofweek >= 5
    out = pd.Series(np.nan, index=s.index)
    win = BASELINE_WEEKS * 7
    for i, day in enumerate(s.index):
        if i < win:
            continue
        past = s.iloc[i - win:i]
        same = past[(past.index.dayofweek >= 5) == is_weekend[i]]
        base = same.median()
        if base > 0:
            out.iloc[i] = s.iloc[i] / base
    return out

def seasonal_adjust(ratio, last_year_ratio=None):
    """작년 같은 날(52주 전 같은 요일) 배수만큼의 반복 증가를 뺌. 작년 기록이 없으면 그대로 씀"""
    if last_year_ratio is None:
        return ratio, False
    ly = last_year_ratio.reindex(ratio.index - pd.Timedelta(weeks=52)).to_numpy()
    adj = ratio - np.clip(np.nan_to_num(ly, nan=1.0) - 1, 0, None)
    return adj.clip(lower=0), True

# ===== 1단계 판정 =====
def judge(ratio, base_date):
    """base_date(판정에 쓰는 마지막 날, 보통 D-1) 기준 1단계 여부"""
    base_date = pd.Timestamp(base_date)
    hist = ratio.loc[base_date - pd.Timedelta(days=HISTORY_DAYS + CONSEC_DAYS - 1):
                     base_date - pd.Timedelta(days=CONSEC_DAYS)].dropna()
    recent = ratio.loc[base_date - pd.Timedelta(days=CONSEC_DAYS - 1):base_date]
    if len(hist) < 60 or recent.isna().any():
        return {"판정": "계산 불가", "이유": "과거 기록 부족 또는 검색량 0"}
    top = np.percentile(hist, 100 - TOP_PCT)
    mx = hist.max()
    over_top = bool((recent >= top).all())
    over_max = bool(recent.iloc[-1] > mx)
    stage1 = over_top or over_max
    why = []
    if over_top: why.append(f"상위 {TOP_PCT:g}%({top:.2f}배) {CONSEC_DAYS}일 연속")
    if over_max: why.append(f"1년 최댓값({mx:.2f}배) 넘음")
    return {"판정": "1단계" if stage1 else "평상", "오늘 배수": round(float(recent.iloc[-1]), 2),
            "상위 기준": round(float(top), 2), "1년 최대": round(float(mx), 2),
            "이유": ", ".join(why) or "-"}

def replay(get_series, base_dates):
    """기준일마다 그날까지 받을 수 있던 자료만 다시 받아서 판정함 (데이터 누수 방지)"""
    rows = []
    for d in base_dates:
        d = pd.Timestamp(d)
        s = get_series(d)          # d까지만 받은 시계열
        assert s.index.max() <= d, "기준일 이후 자료가 섞임"
        res = judge(surge_ratio(s), d)
        rows.append({"자료 기준일(D-1)": d.date(), "판정일 07시": (d + pd.Timedelta(days=1)).date(), **res})
    return pd.DataFrame(rows)
