"""구·군별 데이터랩 검색 지수를 한 번에 받음 (규모 예측 학습용)

  export NAVER_CLIENT_ID=... NAVER_CLIENT_SECRET=...
  python3 -m miribom_ai.scale.collect_search 2024-01-01 2026-10-20

- 검색어는 data/area_keywords.csv (구·군마다 |로 묶음, 최대 20개). 한 구·군당 요청 1번이라 16번이면 끝
- 구·군마다 따로 요청하므로 서로 기준이 섞이지 않음. 값은 구·군 안에서의 상댓값이고, 모델은 배수(평소 대비)만 씀
- 결과: data/raw/search/<구·군>.csv (date, search_index)
"""
import os, sys, time
import pandas as pd
from miribom_ai.paths import ROOT
from miribom_ai.surge.collect import fetch_datalab

def main(start, end):
    cid, cs = os.environ.get("NAVER_CLIENT_ID"), os.environ.get("NAVER_CLIENT_SECRET")
    if not (cid and cs):
        sys.exit("NAVER_CLIENT_ID, NAVER_CLIENT_SECRET 환경변수를 넣을 것")
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    out_dir = ROOT / "data" / "raw" / "search"; out_dir.mkdir(parents=True, exist_ok=True)
    for _, r in pd.read_csv(ROOT / "data" / "area_keywords.csv").iterrows():
        s = fetch_datalab(r["keywords"].split("|")[:20], end, cid, cs, days=days)
        s.rename("search_index").rename_axis("date").reset_index().to_csv(out_dir / f"{r['area']}.csv", index=False)
        print(f"{r['area']}: {len(s)}일, 0인 날 {int((s == 0).sum())}일")
        time.sleep(0.3)

if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "2024-01-01", a[1] if len(a) > 1 else str(pd.Timestamp.today().date() - pd.Timedelta(days=1)))
