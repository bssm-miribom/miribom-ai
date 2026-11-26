"""학습용 과거 날씨 (부산 일별 관측) → data/raw/weather/busan_daily.csv (date, rain, tmax, source)

  python3 -m miribom_ai.scale.collect_weather --start 2023-12-01 --end 2026-10-31

출처 (위에서부터 먼저 있는 것)
  API허브 키        기상청 API허브 종관기상관측(ASOS) 일자료 kma_sfcdd3, 부산 지점(159)
                    (API허브에서 'ASOS 일자료'를 활용신청해야 함)
  공공데이터포털 키  기상청_지상(종관, ASOS) 일자료 조회서비스, 부산 지점(159)
                    (공공데이터포털에서 이 서비스를 따로 활용신청해야 함)
  키는 .env의 KMA_AUTH_KEY 또는 KMA_SERVICE_KEY. 어느 쪽 키인지는 길이로 구분함 (API허브 22자)
  --open-meteo      키 없이 Open-Meteo 과거 자료(ERA5 재분석, 부산 중심 좌표). 시험용
한계
  - 부산 ASOS 지점은 중구 대청동 1곳. 해운대·기장 등 동부는 강수가 다를 수 있음 [팀 확인]
  - 학습에는 '실제 관측'을 씀. 운영에서는 '예보'를 넣으므로, 예보가 틀리는 만큼 실제 개선 폭은 평가보다 작음
"""
import argparse, os
import pandas as pd, requests
from miribom_ai.paths import ROOT

STN = 159                                  # 부산 ASOS
LAT, LON = 35.1047, 129.0320               # 부산 지점 좌표 (Open-Meteo용)
OUT = ROOT / "data" / "raw" / "weather" / "busan_daily.csv"

def _apihub(start, end, key):
    # 응답: '#'로 시작하는 설명 줄 + 공백 구분 표. 1열 TM(YYYYMMDD), 열 순서는 설명 줄 기준
    url = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"
    rows = []
    for y0 in pd.date_range(start, end, freq="YS").union([pd.Timestamp(start)]):
        y1 = min(pd.Timestamp(end), pd.Timestamp(f"{y0.year}-12-31"))
        r = requests.get(url, params=dict(tm1=f"{y0:%Y%m%d}", tm2=f"{y1:%Y%m%d}", stn=STN, help=1, authKey=key), timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"API허브 {r.status_code}: {r.content.decode('utf-8', 'replace')[:120]}")
        rows.append(r.content.decode("euc-kr", errors="replace"))
    return _parse_sfcdd("\n".join(rows))

def _parse_sfcdd(text):
    # help=1이면 열 설명이 '#  n. NAME ...' 형태로 옴. TA_MAX(최고기온), RN_DAY(일강수량)의 열 번호를 찾음
    names = {}
    for line in text.splitlines():
        s = line.lstrip("#").strip()
        parts = s.split()
        if len(parts) >= 2 and parts[0].rstrip(".").isdigit():
            names.setdefault(parts[1], int(parts[0].rstrip(".")) - 1)
    data = [l.split() for l in text.splitlines() if l and not l.startswith("#") and l.split()[0].isdigit()]
    if not data or "TA_MAX" not in names or "RN_DAY" not in names:
        raise RuntimeError("API허브 응답 형식을 못 읽음 (키·활용신청 확인)")
    df = pd.DataFrame({"date": pd.to_datetime([d[0][:8] for d in data]),
                       "tmax": [float(d[names["TA_MAX"]]) for d in data],
                       "rain": [float(d[names["RN_DAY"]]) for d in data]})
    df.loc[df["tmax"] <= -50, "tmax"] = float("nan")
    df.loc[df["rain"] < 0, "rain"] = 0.0             # -9 = 강수 없음
    return df.assign(source="기상청 ASOS(API허브)")

def _datagokr(start, end, key):
    url = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
    rows = []
    for y0 in pd.date_range(start, end, freq="YS").union([pd.Timestamp(start)]):
        y1 = min(pd.Timestamp(end), pd.Timestamp(f"{y0.year}-12-31"))
        p = dict(serviceKey=key, pageNo=1, numOfRows=400, dataType="JSON", dataCd="ASOS", dateCd="DAY",
                 startDt=f"{y0:%Y%m%d}", endDt=f"{y1:%Y%m%d}", stnIds=STN)
        j = requests.get(url, params=p, timeout=60).json()
        hdr = j["response"]["header"]
        if hdr["resultCode"] != "00":
            raise RuntimeError(f"공공데이터포털 오류 {hdr['resultCode']} {hdr['resultMsg']} (ASOS 일자료 활용신청 확인)")
        rows += j["response"]["body"]["items"]["item"]
    df = pd.DataFrame(rows)
    return pd.DataFrame({"date": pd.to_datetime(df["tm"]),
                         "tmax": pd.to_numeric(df["maxTa"], errors="coerce"),
                         "rain": pd.to_numeric(df["sumRn"], errors="coerce").fillna(0.0),   # 빈칸 = 강수 없음
                         "source": "기상청 ASOS(공공데이터포털)"})

def _open_meteo(start, end):
    p = dict(latitude=LAT, longitude=LON, start_date=start, end_date=end, timezone="Asia/Seoul",
             daily="precipitation_sum,temperature_2m_max")
    d = requests.get("https://archive-api.open-meteo.com/v1/archive", params=p, timeout=60).json()["daily"]
    return pd.DataFrame({"date": pd.to_datetime(d["time"]), "tmax": d["temperature_2m_max"],
                         "rain": d["precipitation_sum"], "source": "Open-Meteo(ERA5)"})

def kma_key():
    """(종류, 키). API허브 키는 22자, 공공데이터포털 키는 그보다 훨씬 김. 없으면 (None, None)"""
    key = os.environ.get("KMA_AUTH_KEY") or os.environ.get("KMA_SERVICE_KEY")
    if not key:
        return None, None
    return ("apihub" if len(key) <= 30 else "datagokr"), key

def collect(start, end, open_meteo=False):
    if open_meteo:
        return _open_meteo(start, end)
    kind, key = kma_key()
    if kind == "apihub":
        return _apihub(start, end, key)
    if kind == "datagokr":
        return _datagokr(start, end, key)
    raise RuntimeError("KMA_AUTH_KEY 또는 KMA_SERVICE_KEY가 .env에 없음 (키 없이 시험하려면 --open-meteo)")

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-12-01"); ap.add_argument("--end", default="2026-10-31")
    ap.add_argument("--open-meteo", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    df = collect(a.start, a.end, a.open_meteo).drop_duplicates("date").sort_values("date")
    out = pd.io.common.stringify_path(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{out}: {len(df)}일 {df['date'].min().date()}~{df['date'].max().date()} ({df['source'].iloc[0]}), "
          f"비 온 날(1mm 이상) {int((df['rain'] >= 1).sum())}일, 빈 값 tmax {int(df['tmax'].isna().sum())}")

if __name__ == "__main__":
    main()
