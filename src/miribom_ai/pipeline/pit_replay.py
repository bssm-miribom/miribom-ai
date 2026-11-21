"""시점 고정 되돌려 보기: 판정일마다 그날 D-1까지만 데이터랩을 받아 급증 탐지·경보 판정을 다시 돌림

비교 대상: 장소마다 한 번에 받아 판정일마다 자른 결과 (지금까지의 되돌려 보기 방식)
- 데이터랩은 요청 기간 안의 최댓값을 100으로 맞추므로, 한 번에 받으면 미래 최댓값 기준으로 눌린 값이 됨
- 요청 수를 줄이려고 시점 고정 요청은 아래 날만 함 (전 기간 670일 × 5곳 = 3,350회는 하루 한도 초과 위험)
    1) 한 번에 받은 결과에서 1단계 이상인 날 ± WINDOW일
    2) 장소마다 무작위 대조일 N_CONTROL일 (seed 0)
- 받은 자료는 data/raw/pit/에 저장 (gitignore). 다시 돌리면 저장본을 씀

  python3 -m miribom_ai.pipeline.pit_replay                       # 2025-01-01 ~ 2026-10-31
  python3 -m miribom_ai.pipeline.pit_replay --max-calls 300
"""
import argparse, os, time
import numpy as np, pandas as pd
from miribom_ai.paths import ROOT
from miribom_ai.surge.collect import fetch_datalab
from miribom_ai.surge.config import FETCH_DAYS
from miribom_ai.surge.detect import surge_ratio, judge as judge_surge
from miribom_ai.scale import predict as scale
from miribom_ai.alert.judge import judge as judge_alert
from miribom_ai.pipeline.holiday_calendar import holiday_set

WINDOW = 7
N_CONTROL = 20
PIT = ROOT / "data" / "raw" / "pit"

def cached(path, fetch):
    if path.exists():
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        return s, False
    s = fetch(); path.parent.mkdir(parents=True, exist_ok=True)
    s.rename("ratio").to_csv(path); return s, True

def stage(s, base, hol, p):
    """run_daily와 같은 순서: 급증 탐지 → (1단계면) 규모 예측 → 경보 판정"""
    sj = judge_surge(surge_ratio(s), base)
    out = {"급증": sj["판정"], "D-1 배수": sj.get("오늘 배수")}
    if sj["판정"] != "1단계" and int(p.get("force_watch", 0) or 0) != 1:
        return {**out, "단계": "평상"}
    if pd.isna(p["base_rest"]) or pd.isna(p["base_work"]):
        return {**out, "단계": "판정 보류"}
    pred, _ = scale.predict_place(s, base, hol, p["base_rest"], p["base_work"])
    return {**out, "단계": judge_alert(pred, p.to_dict())["단계"]}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01"); ap.add_argument("--end", default="2026-10-31")
    ap.add_argument("--max-calls", type=int, default=600)
    a = ap.parse_args(argv)
    cid, cs = os.environ["NAVER_CLIENT_ID"], os.environ["NAVER_CLIENT_SECRET"]
    D0, D1 = pd.Timestamp(a.start), pd.Timestamp(a.end)
    days = pd.date_range(D0, D1)
    places = pd.read_csv(ROOT / "places.csv")
    hol = holiday_set(D0 - pd.Timedelta(days=600), D1 + pd.Timedelta(days=10))
    scale.use_demo(False)
    calls = 0

    # 1) 한 번에 받아 자른 결과 (전 기간)
    one = []
    for _, p in places.iterrows():
        kw = p["keywords"].split("|")[:20]
        full, new = cached(PIT / f"_full_{p['place']}_{D1.date()}.csv",
                           lambda: fetch_datalab(kw, D1 - pd.Timedelta(days=1), cid, cs, days=(D1 - D0).days + FETCH_DAYS))
        calls += new
        for jd in days:
            base = jd - pd.Timedelta(days=1)
            one.append({"장소": p["place"], "판정일": jd, **stage(full[full.index <= base], base, hol, p)})
    one = pd.DataFrame(one)

    # 2) 시점 고정으로 다시 볼 날 고르기
    rng = np.random.default_rng(0)
    pick = set()
    for pl, g in one.groupby("장소"):
        hot = g.loc[g["단계"] != "평상", "판정일"]
        for d in hot:
            for k in range(-WINDOW, WINDOW + 1):
                if D0 <= d + pd.Timedelta(days=k) <= D1: pick.add((pl, d + pd.Timedelta(days=k)))
        for d in rng.choice(days, N_CONTROL, replace=False):
            pick.add((pl, pd.Timestamp(d)))
    pick = sorted(pick)
    print(f"시점 고정 판정 {len(pick)}건 (저장본 없으면 각 1회 요청, 한도 {a.max_calls}회)")

    # 3) 시점 고정 판정
    rows, skipped = [], 0
    pmap = {p["place"]: p for _, p in places.iterrows()}
    for pl, jd in pick:
        p, base = pmap[pl], jd - pd.Timedelta(days=1)
        path = PIT / pl / f"{base.date()}.csv"
        if not path.exists() and calls >= a.max_calls:
            skipped += 1; continue
        kw = p["keywords"].split("|")[:20]
        s, new = cached(path, lambda: fetch_datalab(kw, base, cid, cs))
        if new: calls += 1; time.sleep(0.2)
        assert s.index.max() <= base, "기준일 이후 자료가 섞임"
        rows.append({"장소": pl, "판정일": jd, **{f"시점고정 {k}": v for k, v in stage(s, base, hol, p).items()}})
    pit = pd.DataFrame(rows)
    m = one.merge(pit, on=["장소", "판정일"], how="inner")
    m["같음"] = m["단계"] == m["시점고정 단계"]
    m.to_csv(ROOT / "results" / "pit_replay.csv", index=False, encoding="utf-8-sig")

    print(f"API 요청 {calls}회, 한도로 건너뜀 {skipped}건")
    print(f"비교 {len(m)}건 중 단계가 다른 날 {(~m['같음']).sum()}건, 급증 판정이 다른 날 {(m['급증'] != m['시점고정 급증']).sum()}건")
    diff = m[~m["같음"] | (m["급증"] != m["시점고정 급증"])]
    if len(diff):
        print(diff[["장소", "판정일", "D-1 배수", "급증", "단계", "시점고정 D-1 배수", "시점고정 급증", "시점고정 단계"]].to_string(index=False))
    for col in ("단계", "시점고정 단계"):
        hi = m[m[col].isin(["2단계", "3단계"])]
        print(f"{col} 2단계 이상:", hi.groupby("장소")["판정일"].agg(["min", "max", "count"]).to_dict("index"))
    hi1 = one[one["단계"].isin(["2단계", "3단계"])]
    print("한 번에 받은 결과 전 기간 2단계 이상:", hi1.groupby("장소")["판정일"].agg(["min", "max", "count"]).to_dict("index"))

if __name__ == "__main__":
    main()
