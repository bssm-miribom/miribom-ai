"""판정일 D 아침 기준 D+1~3 날씨 예보 (일 강수량 mm, 최고기온). 학습 입력(rain, tmax)과 같은 뜻으로 맞춤

- 기상청 단기예보(VilageFcstInfoService_2.0, 05시 발표): API허브 키 또는 공공데이터포털 키 (collect_weather.kma_key)
  강수량 = 시간별 PCP 합, 최고기온 = TMX
- 키가 없거나 실패하면 Open-Meteo 예보 (키 필요 없음, 비상업 무료. 시험용)
- 둘 다 실패하면 빈 값. 모델은 빈 값을 '날씨 모름'으로 보고 예측함
"""
import math, re
import pandas as pd, requests
from miribom_ai.scale.collect_weather import kma_key, LAT, LON

KMA_URLS = {"apihub": "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst",
            "datagokr": "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"}

def latlon_to_grid(lat, lon):
    """위경도 → 기상청 격자(nx, ny). 기상청 공개 변환식(LCC)"""
    RE, GRID, SLAT1, SLAT2, OLON, OLAT, XO, YO = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0, 43, 136
    DEG = math.pi / 180.0
    re_ = RE / GRID
    slat1, slat2, olon, olat = SLAT1 * DEG, SLAT2 * DEG, OLON * DEG, OLAT * DEG
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5))
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5) ** sn * math.cos(slat1) / sn
    ro = re_ * sf / math.tan(math.pi * 0.25 + olat * 0.5) ** sn
    ra = re_ * sf / math.tan(math.pi * 0.25 + lat * DEG * 0.5) ** sn
    theta = lon * DEG - olon
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    theta *= sn
    return int(ra * math.sin(theta) + XO + 0.5), int(ro - ra * math.cos(theta) + YO + 0.5)

def pcp_mm(v):
    """단기예보 PCP 글자 → mm. '강수없음'=0, '1mm 미만'=0.5, '30.0~50.0mm'=40, '50.0mm 이상'=50"""
    v = str(v).strip()
    if v in ("강수없음", "-", "0", ""):
        return 0.0
    if "미만" in v:
        return 0.5
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", v)]
    return sum(nums) / len(nums) if nums else 0.0

def _kma(lat, lon, day, kind, key):
    nx, ny = latlon_to_grid(lat, lon)
    p = dict(pageNo=1, numOfRows=1500, dataType="JSON", base_date=f"{day:%Y%m%d}", base_time="0500", nx=nx, ny=ny)
    p["authKey" if kind == "apihub" else "serviceKey"] = key
    r = requests.get(KMA_URLS[kind], params=p, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"기상청 {r.status_code}: {r.content.decode('utf-8', 'replace')[:100]}")
    items = r.json()["response"]["body"]["items"]["item"]
    df = pd.DataFrame(items)
    df["date"] = pd.to_datetime(df["fcstDate"])
    rain = df[df["category"] == "PCP"].assign(v=lambda x: x["fcstValue"].map(pcp_mm)).groupby("date")["v"].sum()
    tmax = pd.to_numeric(df[df["category"] == "TMX"].set_index("date")["fcstValue"], errors="coerce").groupby(level=0).max()
    return rain, tmax

def _open_meteo(lat, lon):
    p = dict(latitude=lat, longitude=lon, timezone="Asia/Seoul", forecast_days=5,
             daily="precipitation_sum,temperature_2m_max")
    d = requests.get("https://api.open-meteo.com/v1/forecast", params=p, timeout=30).json()["daily"]
    idx = pd.to_datetime(d["time"])
    return pd.Series(d["precipitation_sum"], index=idx, dtype=float), pd.Series(d["temperature_2m_max"], index=idx, dtype=float)

def forecast(judge_day, lat=LAT, lon=LON):
    """판정일 D 05시 발표 예보로 {rain_h1, tmax_h1, ..., rain_h3, tmax_h3}와 출처를 돌려줌
    학습이 부산 지점(159) 1곳 관측이라 기본값도 그 좌표를 씀"""
    day = pd.Timestamp(judge_day).normalize()
    kind, key = kma_key()
    src, err = None, None
    if kind:
        try:
            rain, tmax = _kma(lat, lon, day, kind, key); src = "기상청 단기예보"
        except Exception as e:
            err = f"기상청 실패({' '.join(str(e).split())[:60]})"
    if src is None:
        try:
            rain, tmax = _open_meteo(lat, lon); src = "Open-Meteo" + (f" ({err})" if err else "")
        except Exception as e:
            return {}, f"날씨 실패: {str(e)[:80]}"
    out = {}
    for h in (1, 2, 3):
        t = day + pd.Timedelta(days=h)
        out[f"rain_h{h}"] = float(rain.get(t, float("nan")))
        out[f"tmax_h{h}"] = float(tmax.get(t, float("nan")))
    return out, src
