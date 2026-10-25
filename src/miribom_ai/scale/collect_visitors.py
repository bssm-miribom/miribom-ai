"""부산 구·군별 일별 방문자 수 수집 (규모 예측의 정답)

출처: 공공데이터포털 '한국관광공사_빅데이터_지역별 방문자수_GW' (data.go.kr/data/15101972)
  - KT 통신 자료(내국인)와 SKT 자료(외국인)로 만든 기초지자체별 일별 방문자 수
  - 개발 단계 자동 승인, 하루 1,000회
  - 광역·기초 집계 방식이 달라 구·군 값을 더해도 부산 전체와 맞지 않음

  export TOUR_SERVICE_KEY=공공데이터포털_인증키(Decoding)
  python3 -m miribom_ai.scale.collect_visitors 2024-01-01 2026-10-20

오퍼레이션 이름·응답 필드는 공식 명세(활용가이드)와 다르면 아래 상수만 고치면 됨 [팀 확인]
"""
import os, sys, time
import pandas as pd, requests
from miribom_ai.paths import ROOT

URL = "https://apis.data.go.kr/B551011/DataLabService/locgoRegnVisitrDDList"
BUSAN_PREFIX = "26"            # 부산 시군구 코드 앞자리

def fetch_month(start, end, key):
    rows, page = [], 1
    while True:
        params = dict(serviceKey=key, MobileOS="ETC", MobileApp="miribom", _type="json",
                      startYmd=f"{start:%Y%m%d}", endYmd=f"{end:%Y%m%d}", numOfRows=10000, pageNo=page)
        r = requests.get(URL, params=params, timeout=60)
        r.raise_for_status()
        body = r.json()["response"]["body"]
        items = body.get("items") or {}
        items = items.get("item", []) if isinstance(items, dict) else []
        rows += items if isinstance(items, list) else [items]
        if page * 10000 >= int(body.get("totalCount", 0)):
            return rows
        page += 1

def collect(start, end, key):
    out = []
    for m0 in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        m1 = min(m0 + pd.offsets.MonthEnd(0), pd.Timestamp(end))
        s = max(m0, pd.Timestamp(start))
        out += fetch_month(s, m1, key)
        print(f"{s:%Y-%m} 받음 (누적 {len(out):,}줄)")
        time.sleep(0.3)
    df = pd.DataFrame(out)
    df = df[df["signguCode"].astype(str).str.startswith(BUSAN_PREFIX)]
    df["date"] = pd.to_datetime(df["baseYmd"].astype(str))
    df["touNum"] = pd.to_numeric(df["touNum"], errors="coerce")
    # 현지인·외지인·외국인을 나눠서 받고, visitors는 셋의 합으로 둠
    wide = df.pivot_table(index=["date", "signguNm"], columns="touDivNm", values="touNum", aggfunc="sum")
    wide["visitors"] = wide.sum(axis=1)
    return wide.reset_index().rename(columns={"signguNm": "area"})

if __name__ == "__main__":
    key = os.environ.get("TOUR_SERVICE_KEY")
    if not key:
        sys.exit("TOUR_SERVICE_KEY 환경변수에 공공데이터포털 인증키를 넣을 것")
    start, end = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2024-01-01", str(pd.Timestamp.today().date()))
    df = collect(start, end, key)
    out = ROOT / "data" / "raw" / "visitors_busan.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out} ({df['area'].nunique()}개 구·군, {df['date'].min().date()}~{df['date'].max().date()})")
