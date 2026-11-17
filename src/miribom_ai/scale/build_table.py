"""학습용 표(data/scale_table.csv) 만들기: 방문자 수 + 데이터랩 검색 지수 + 공휴일

준비물
  data/raw/visitors_busan.csv        collect_visitors.py로 받은 구·군별 일별 방문자 수
  data/raw/search/<구·군>.csv        collect_search.py로 받은 검색 지수 (API 키 있을 때)
  또는 data/raw/datalab/<구·군>.csv  데이터랩 웹에서 직접 받은 일간 CSV (파일 이름 = 구·군 이름, 예: 해운대구.csv)

  python3 -m miribom_ai.scale.build_table
"""
import pandas as pd
from miribom_ai.paths import ROOT
from miribom_ai.surge.collect import load_datalab_csv
from miribom_ai.pipeline.holiday_calendar import holiday_set

def main():
    raw = ROOT / "data" / "raw"
    vis = pd.read_csv(raw / "visitors_busan.csv", parse_dates=["date"])[["date", "area", "visitors"]]
    parts, missing = [], []
    for area in sorted(vis["area"].unique()):
        f_api, f_web = raw / "search" / f"{area}.csv", raw / "datalab" / f"{area}.csv"
        if f_api.exists():
            s = pd.read_csv(f_api, parse_dates=["date"])
        elif f_web.exists():
            s = load_datalab_csv(f_web).rename("search_index").rename_axis("date").reset_index()
        else:
            missing.append(area); continue
        parts.append(vis[vis["area"] == area].merge(s, on="date", how="inner"))
    if missing:
        print("검색 지수 파일이 없어서 뺀 구·군:", ", ".join(missing))
    df = pd.concat(parts, ignore_index=True)
    hol = holiday_set(df["date"].min(), df["date"].max())
    df["holiday"] = df["date"].isin(hol).astype(int)
    out = ROOT / "data" / "scale_table.csv"
    df.sort_values(["area", "date"]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out} ({df['area'].nunique()}개 구·군, {len(df):,}줄, {df['date'].min().date()}~{df['date'].max().date()})")

if __name__ == "__main__":
    main()
