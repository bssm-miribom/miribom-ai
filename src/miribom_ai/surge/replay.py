"""과거 사례 재현: 기준일마다 그날까지 자료만으로 1단계 판정을 다시 돌림 (부캉이 재현용)

사용 예
  python3 -m miribom_ai.surge.replay --fake                                   # 가짜 데이터로 연습 (키 필요 없음)
  python3 -m miribom_ai.surge.replay --csv datalab_북항친수공원.csv            # 데이터랩 웹 CSV로 재현
  python3 -m miribom_ai.surge.replay --keywords 북항친수공원                   # 네이버 API로 재현 (환경변수에 키 필요)
  python3 -m miribom_ai.surge.replay --csv 파일.csv --plot                     # 추이 그래프도 저장
"""
import argparse, os
import numpy as np, pandas as pd
from miribom_ai.surge.collect import fetch_datalab, load_datalab_csv
from miribom_ai.surge.detect import surge_ratio, replay
from miribom_ai.surge.config import TOP_PCT

def fake_series():
    """9.18. 등장 후 급증하는 가짜 시계열 (실제 값 아님)"""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", "2026-09-30")
    s = pd.Series((10 + 4 * (idx.dayofweek >= 5) + rng.normal(0, 1.5, len(idx))).clip(1), index=idx)
    for d, m in zip(pd.date_range("2026-09-18", "2026-09-30"), [1.5, 4, 8, 15, 25, 30, 40, 35, 30, 25, 20, 15, 10]):
        s[d] *= m
    return s

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fake", action="store_true", help="가짜 데이터로 연습")
    src.add_argument("--csv", help="데이터랩 웹에서 내려받은 CSV 경로")
    src.add_argument("--keywords", help="API로 받을 검색어. 여러 개는 | 로 구분")
    ap.add_argument("--start", default="2026-09-15", help="첫 기준일(D-1)")
    ap.add_argument("--end", default="2026-09-23", help="마지막 기준일(D-1)")
    ap.add_argument("--plot", action="store_true", help="추이 그래프를 PNG로 저장")
    a = ap.parse_args()

    if a.keywords:
        cid, cs = os.environ["NAVER_CLIENT_ID"], os.environ["NAVER_CLIENT_SECRET"]
        kws = a.keywords.split("|")
        get = lambda d: fetch_datalab(kws, d, cid, cs)   # 기준일마다 그날까지만 요청
    else:
        full = fake_series() if a.fake else load_datalab_csv(a.csv)
        # CSV 하나로도 됨. 데이터랩 값은 전체에 같은 배율이 곱해진 값이라 배수·순위가 변하지 않음
        get = lambda d: (full.loc[:d] / full.loc[:d].max() * 100)

    dates = pd.date_range(a.start, a.end)
    result = replay(get, dates)
    pd.set_option("display.width", 200)
    print(result.to_string(index=False))
    out = f"data/replay_{dates[0]:%Y%m%d}_{dates[-1]:%Y%m%d}.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = get(dates[-1])
        r = surge_ratio(s)
        hist = r.loc[:dates[0]].dropna().iloc[-365:]
        ax = r.loc[dates[0] - pd.Timedelta(days=60):].plot(figsize=(10, 4), label="surge ratio")
        ax.axhline(np.percentile(hist, 100 - TOP_PCT), ls="--", c="r", label=f"top {TOP_PCT:g}% (1y)")
        ax.legend()
        png = out.replace(".csv", ".png")
        plt.savefig(png, dpi=150, bbox_inches="tight")
        print(f"그래프: {png}")

if __name__ == "__main__":
    main()
