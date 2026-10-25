"""데이터랩 검색 지수 수집 (API 또는 웹 CSV)"""
import json
import pandas as pd, requests
from miribom_ai.surge.config import DATALAB_URL, FETCH_DAYS

def fetch_datalab(keywords, end_date, client_id, client_secret, days=FETCH_DAYS):
    """장소 1곳(검색어 묶음 1개)을 end_date까지만 요청함. 기준일 이후 날짜는 절대 요청하지 않음"""
    end = pd.Timestamp(end_date)
    start = end - pd.Timedelta(days=days - 1)
    body = {
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "timeUnit": "date",
        "keywordGroups": [{"groupName": keywords[0], "keywords": keywords}],
    }
    # API HUB 헤더 (예전 개발자센터의 X-Naver-Client-Id 방식은 신규 발급이 막힘)
    headers = {"X-NCP-APIGW-API-KEY-ID": client_id, "X-NCP-APIGW-API-KEY": client_secret,
               "Content-Type": "application/json"}
    r = requests.post(DATALAB_URL, headers=headers, data=json.dumps(body), timeout=30)
    r.raise_for_status()
    rows = r.json()["results"][0]["data"]
    s = pd.Series({pd.Timestamp(x["period"]): x["ratio"] for x in rows}, dtype=float)
    # 데이터랩은 검색량이 0인 날을 빼고 주므로 0으로 채움
    return s.reindex(pd.date_range(start, end), fill_value=0.0)

def load_datalab_csv(path):
    """데이터랩 웹사이트에서 내려받은 CSV로 대신할 때 (API 키가 없을 때)"""
    raw = pd.read_csv(path, skiprows=6, header=None, names=["date", "ratio"], encoding="utf-8")
    raw["date"] = pd.to_datetime(raw["date"].astype(str).str.strip().str.replace(".", "-", regex=False), format="%Y-%m-%d", errors="coerce")
    raw = raw.dropna(subset=["date"])
    s = raw.set_index("date")["ratio"].astype(float)
    return s.reindex(pd.date_range(s.index.min(), s.index.max()), fill_value=0.0)
