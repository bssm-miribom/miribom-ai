"""피크 1시간 비중 계산: 도시철도 시간대별 하차로 장소별 '하루 중 가장 붐빈 1시간' 비중을 구함 → places.csv peak_share

  python3 -m miribom_ai.alert.peak_share
입력: data/raw/metro/busan_metro_hourly.csv (data.go.kr 3057229, 커밋 안 함)
값: 휴일(주말) 기준 상위 10% (p90). 역↔장소 대응은 scale.build_place_table.STATIONS와 같은 기준 [팀 확인]
"""
import pandas as pd
from miribom_ai.paths import ROOT

STATIONS = {"북항친수공원": ["중앙", "부산역", "초량"], "광안리해수욕장": ["광안", "금련산"],
            "해운대해수욕장": ["해운대", "중동"], "감천문화마을": ["토성"], "흰여울문화마을": ["남포"]}

def main():
    h = pd.read_csv(ROOT / "data" / "raw" / "metro" / "busan_metro_hourly.csv", parse_dates=["date"])
    h = h[h["type"] == "하차"]
    cols = [c for c in h.columns if "시-" in c]
    for place, st in STATIONS.items():
        g = h[h["station"].isin(st)].groupby("date")[cols].sum()
        share = g.max(axis=1) / g.sum(axis=1)
        rest = g.index.dayofweek >= 5
        print(f"{place:<8} 휴일 중앙값 {share[rest].median():.3f}  p90 {share[rest].quantile(.9):.3f}  "
              f"평일 중앙값 {share[~rest].median():.3f}  최대 {share.max():.3f}")

if __name__ == "__main__":
    main()
