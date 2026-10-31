"""장소 단위 학습 표 만들기: 장소 검색 지수 + 가까운 도시철도역 하차 인원(방문객 대리 지표) → data/place_table.csv

  python3 -m miribom_ai.scale.build_place_table
  python3 -m miribom_ai.scale.train --table data/place_table.csv [--mode learned]

입력 (data/raw/, 커밋 안 함)
  places/<장소>.csv              네이버 데이터랩 일별 검색 지수 (date, ratio)
  metro/busan_metro_daily.csv    부산교통공사_시간대별 승하차인원 (data.go.kr 3057229)을 일별로 합친 것
한계
  - 역 하차 인원에는 출퇴근·환승객이 섞여 있어 장소 방문 증가가 실제보다 작게 보임 (대리 지표)
  - 영도(흰여울·태종대)·송정·기장처럼 도시철도가 없는 곳은 못 씀
  - 장소↔역 대응은 걸어서 가는 가장 가까운 역 기준 [팀 확인]
"""
import pandas as pd
from miribom_ai.paths import ROOT
from miribom_ai.pipeline.holiday_calendar import holiday_set

STATIONS = {
    "광안리": ["광안", "금련산"], "광안리드론쇼": ["광안", "금련산"], "부산불꽃축제": ["광안", "금련산"],
    "해운대": ["해운대", "중동"], "해운대빛축제": ["해운대", "중동"], "블루라인파크": ["해운대", "중동"],
    "다대포해수욕장": ["다대포해수욕장"], "자갈치시장": ["자갈치", "남포"], "국제시장": ["자갈치", "남포"],
    "BIFF광장": ["자갈치", "남포"], "용두산공원": ["남포", "중앙"], "서면": ["서면"], "전포카페거리": ["전포"],
    "감천문화마을": ["토성"], "범어사": ["범어사"], "부산국제영화제": ["센텀시티"], "부산시민공원": ["부전"],
    "북항친수공원": ["중앙", "부산역", "초량"],   # 부캉이(2026.9.)는 train.py --exclude로 학습에서 빠짐
}

def main():
    raw = ROOT / "data" / "raw"
    m = pd.read_csv(raw / "metro" / "busan_metro_daily.csv", parse_dates=["date"])
    alight = m.groupby(["date", "station"])["alight"].sum().unstack()
    hol = holiday_set(alight.index.min() - pd.Timedelta(days=30), alight.index.max() + pd.Timedelta(days=30))
    out = []
    for place, st in STATIONS.items():
        f = raw / "places" / f"{place}.csv"
        if not f.exists():
            print("검색 지수 없음:", place); continue
        q = pd.read_csv(f, parse_dates=["date"]).set_index("date")["ratio"]
        v = alight[st].sum(axis=1)
        df = pd.DataFrame({"date": v.index, "area": place, "visitors": v.values,
                           "search_index": q.reindex(v.index).values})
        df["holiday"] = [int(d in hol) for d in df["date"]]
        out.append(df.dropna(subset=["search_index"]))
    t = pd.concat(out)
    t.to_csv(ROOT / "data" / "place_table.csv", index=False)
    print(f"data/place_table.csv: {t['area'].nunique()}곳, {len(t)}줄, {t['date'].min().date()}~{t['date'].max().date()}")

if __name__ == "__main__":
    main()
