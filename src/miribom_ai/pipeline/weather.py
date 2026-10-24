"""D+1~3 날씨 예보 (강수확률, 최고기온)

- KMA_SERVICE_KEY(공공데이터포털 기상청 단기예보 키)가 있으면 기상청 단기예보를 씀 (운영용)
- 없으면 Open-Meteo를 씀 (키 필요 없음, 비상업 무료. 시제품·시연용)
- 둘 다 실패하면 빈 값으로 두고, 모델은 날씨 없이 예측함
"""
import math, os
import pandas as pd, requests

KMA_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

def latlon_to_grid(lat, lon):
    """위경도 → 기상청 격자(nx, ny). 기상청 공개 변환식(LCC)"""
    RE, GRID, SLAT1, SLAT2, OLON, OLAT, XO, YO = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0, 43, 136
    DEG = math.pi / 180.0
    re = RE / GRID
    slat1, slat2, olon, olat = SLAT1 * DEG, SLAT2 * DEG, OLON * DEG, OLAT * DEG
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5))
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5) ** sn * math.cos(slat1) / sn
    ro = re * sf / math.tan(math.pi * 0.25 + olat * 0.5) ** sn
    ra = re * sf / math.tan(math.pi * 0.25 + lat * DEG * 0.5) ** sn
    theta = lon * DEG - olon
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    theta *= sn
    return int(ra * math.sin(theta) + XO + 0.5), int(ro - ra * math.cos(theta) + YO + 0.5)

def _kma(lat, lon, day, key):
    nx, ny = latlon_to_grid(lat, lon)
    params = dict(serviceKey=key, pageNo=1, numOfRows=1500, dataType="JSON",
                  base_date=f"{day:%Y%m%d}", base_time="0500", nx=nx, ny=ny)
    items = requests.get(KMA_URL, params=params, timeout=30).json()["response"]["body"]["items"]["item"]
    df = pd.DataFrame(items)
    df["date"] = pd.to_datetime(df["fcstDate"])
    df["fcstValue"] = pd.to_numeric(df["fcstValue"], errors="coerce")
    rain = df[df["category"] == "POP"].groupby("date")["fcstValue"].max()
    tmax = df[df["category"] == "TMX"].groupby("date")["fcstValue"].max()
    return rain, tmax

def _open_meteo(lat, lon):
    params = dict(latitude=lat, longitude=lon, timezone="Asia/Seoul", forecast_days=5,
                  daily="precipitation_probability_max,temperature_2m_max")
    d = requests.get(OPEN_METEO_URL, params=params, timeout=30).json()["daily"]
    idx = pd.to_datetime(d["time"])
    return (pd.Series(d["precipitation_probability_max"], index=idx, dtype=float),
            pd.Series(d["temperature_2m_max"], index=idx, dtype=float))

def forecast(lat, lon, judge_day):
    """판정일 D 아침 기준 D+1~3 예보를 {rain_h1: .., tmax_h1: ..} 형태로 돌려줌"""
    out = {}
    if pd.isna(lat) or pd.isna(lon):
        return out, "좌표 없음"
    day = pd.Timestamp(judge_day)
    try:
        key = os.environ.get("KMA_SERVICE_KEY")
        rain, tmax = _kma(lat, lon, day, key) if key else _open_meteo(lat, lon)
        src = "기상청" if key else "Open-Meteo"
    except Exception as e:
        return out, f"날씨 실패: {str(e)[:80]}"
    for h in (1, 2, 3):
        t = day + pd.Timedelta(days=h)
        out[f"rain_h{h}"] = float(rain.get(t, float("nan")))
        out[f"tmax_h{h}"] = float(tmax.get(t, float("nan")))
    return out, src
