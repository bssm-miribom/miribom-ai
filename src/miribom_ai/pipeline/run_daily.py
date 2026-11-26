"""하루 판정 전체 흐름: 수집 → 급증 탐지 → 규모 예측 → 경보 판정 → 결과(dict) 반환
FastAPI 서버에서는 miribom_ai.run_daily()로 불러서 씀. 예약 실행·알림 발송은 서버가 맡음

  python3 -m miribom_ai.pipeline.run_daily --demo            # 키 없이 가짜 데이터로 전체 흐름 시험 (2026.9.20. 판정)
  python3 -m miribom_ai.pipeline.run_daily                   # 오늘 판정 (네이버 키 필요)
  python3 -m miribom_ai.pipeline.run_daily --date 2026-09-20 # 특정 판정일

필요한 환경변수: NAVER_CLIENT_ID, NAVER_CLIENT_SECRET (필수, --demo 제외)
선택: KMA_SERVICE_KEY 또는 KMA_AUTH_KEY (기상청 단기예보, 모델이 날씨를 쓸 때만. 없으면 Open-Meteo)
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, pandas as pd
from miribom_ai.surge.collect import fetch_datalab
from miribom_ai.surge.detect import surge_ratio, judge as judge_surge
from miribom_ai.scale import predict as scale
from miribom_ai.alert.judge import judge as judge_alert
from miribom_ai.pipeline.holiday_calendar import holiday_set
from miribom_ai.pipeline.weather import forecast

from miribom_ai.paths import ROOT

def fake_search(place, base):
    """--demo용 가짜 검색 지수. 북항친수공원만 9.18.부터 급증 (실제 값 아님)"""
    rng = np.random.default_rng(abs(hash(place)) % 2**32)
    idx = pd.date_range(base - pd.Timedelta(days=450), base)
    s = pd.Series((10 + 4 * (idx.dayofweek >= 5) + rng.normal(0, 1.5, len(idx))).clip(1), index=idx)
    if place == "북항친수공원":
        for d, m in zip(pd.date_range("2026-09-18", "2026-09-30"), [1.5, 4, 8, 15, 25, 30, 40, 35, 30, 25, 20, 15, 10]):
            if d in s.index: s[d] *= m
    return s / s.max() * 100

def intent_series(place):
    """방문 의도 분류 결과가 쌓여 있으면 장소별 하루 가볼래 글 수를 씀 (Threads 승인 후)"""
    p = ROOT / "data" / "intent_counts.csv"
    if not p.exists():
        return None
    c = pd.read_csv(p, parse_dates=["date"])
    c = c[c["place"] == place]
    return c.groupby("date")["intent_posts"].sum() if len(c) else None

def intent_summary(series, base):
    """자료 기준일(D-1) 가볼래 글 수와 평소(직전 28일 중앙값) 대비 배수. 자료 없으면 None"""
    if series is None or base not in series.index:
        return None
    prev = series[(series.index < base) & (series.index >= base - pd.Timedelta(days=28))]
    med = float(prev.median()) if len(prev) else float("nan")
    return {"어제 글 수": int(series[base]),
            "평소 대비": round(float(series[base]) / med, 2) if med and med > 0 else None}

def nan_to_none(x):
    if isinstance(x, dict): return {k: nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list): return [nan_to_none(v) for v in x]
    if isinstance(x, float) and np.isnan(x): return None
    return x

class ModelMissing(RuntimeError):
    pass

def apply_scenario(s, place, scenario):
    """가상 시나리오: 한 장소의 검색 지수를 start일부터 배수만큼 키움 (실제 값 아님, 위험 단계 시연용).
    나머지 장소·과거 값은 그대로라 같은 모델·판정 규칙이 커진 입력에 어떻게 반응하는지 보여줌"""
    if not scenario or place != scenario["place"]:
        return s
    s = s.astype(float).copy()
    s[s.index >= pd.Timestamp(scenario["start"])] *= float(scenario["multiplier"])
    return s

SCENARIO_ASSUME = "검색이 늘어난 만큼 방문도 늘어난다고 가정 (최악 시나리오 값을 예측값으로 씀, 실측으로 확인 안 된 가정)"

def scenario_pred(pred):
    """가상 시나리오의 예측: 실측 모델은 검색량에 거의 반응하지 않아서, '최악 시나리오'(검색 N배면 방문 N배)를 예측값으로 씀"""
    out = {}
    for h, v in pred.items():
        w = v.get("최악")
        out[h] = v if w is None else {**v, "하위10": v["중앙값"], "중앙값": w, "상위90": w}
    return out

def run(judge_day=None, demo=False, save=True, scenario=None):
    """judge_day: 판정일 D (기본 오늘, 한국시간). save=True면 data/daily/에 JSON도 저장
    scenario: {"place": 장소 이름, "multiplier": 배수, "start": "YYYY-MM-DD"} 가상 시나리오 (이때는 저장 안 함).
      그 장소만 검색 지수를 start일부터 배수만큼 키우고, 예측은 '검색 증가 = 방문 증가' 가정(SCENARIO_ASSUME)으로 바꿈"""
    if scenario:
        save = False
    if judge_day is None:
        judge_day = "2026-09-20" if demo else pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
    judge_day = pd.Timestamp(judge_day)
    base = judge_day - pd.Timedelta(days=1)                       # 자료는 D-1까지
    scale.use_demo(demo)
    if not scale.models_ready():
        if not demo:   # 운영에서 가짜 데이터 모델로 예측하면 안 됨
            raise ModelMissing("models/에 규모 예측 모델이 없음. 실제 데이터로 python3 -m miribom_ai.scale.train 실행 후 models/를 커밋할 것")
        print("규모 예측 모델이 없어서 가짜 데이터로 먼저 학습함 (실제 학습은 python3 -m miribom_ai.scale.train)")
        from miribom_ai.scale.train import main as train_main
        train_main(["--demo"])
    hol = holiday_set(base - pd.Timedelta(days=500), judge_day + pd.Timedelta(days=10))
    if not demo:
        cid, cs = os.environ.get("NAVER_CLIENT_ID"), os.environ.get("NAVER_CLIENT_SECRET")
        if not (cid and cs):
            raise RuntimeError("NAVER_CLIENT_ID, NAVER_CLIENT_SECRET 환경변수가 없음")
    places = pd.read_csv(ROOT / "places.csv")
    results = []
    weather, wsrc = None, None
    for _, p in places.iterrows():
        r = {"장소": p["place"], "판정일": str(judge_day.date()), "자료 기준일": str(base.date())}
        try:
            s = fake_search(p["place"], base) if demo else fetch_datalab(p["keywords"].split("|")[:20], base, cid, cs)
            s = apply_scenario(s, p["place"], scenario)
            sj = judge_surge(surge_ratio(s), base)
            r["급증"] = sj
            intent = intent_series(p["place"])
            if intent is not None:
                intent = intent[intent.index <= base]             # 판정일 이후 글은 쓰지 않음 (누수 방지)
            r["의도"] = intent_summary(intent, base)
            if sj["판정"] != "1단계" and int(p.get("force_watch", 0) or 0) != 1:
                r["단계"], r["이유"] = "평상", sj.get("이유", "-")
                results.append(r); continue
            if pd.isna(p["base_rest"]) or pd.isna(p["base_work"]):
                r["단계"], r["이유"] = "판정 보류", "평소 방문객 값 없음 (places.csv)"
                r["빠진 값"] = [k for k in ("base_rest", "base_work") if pd.isna(p[k])]
                results.append(r); continue
            if weather is None:                                   # 부산 지점 1곳 예보라 판정일마다 한 번만 받음
                weather, wsrc = forecast(judge_day) if scale.uses_weather() else ({}, "모델이 날씨를 안 씀")
            pred, _ = scale.predict_place(s, base, hol, p["base_rest"], p["base_work"], intent=intent, weather=weather)
            if scenario and p["place"] == scenario["place"]:
                r["실측 모델 예측"], r["가정"] = pred, SCENARIO_ASSUME
                pred = scenario_pred(pred)
            r["예측"], r["날씨 출처"] = pred, wsrc
            r.update(judge_alert(pred, p.to_dict()))
        except Exception as e:                                    # 한 곳이 실패해도 나머지는 계속
            r["단계"], r["이유"] = "수집 실패", f"{type(e).__name__}: {str(e)[:200]}"
        results.append(r)
        if not demo: time.sleep(0.2)

    payload = nan_to_none({"판정일": str(judge_day.date()), "생성 시각": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                           "가짜 데이터": demo, "가상 시나리오": scenario, "장소": results})
    if save:
        out_dir = ROOT / "data" / "daily"; out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{judge_day:%Y%m%d}{'_demo' if demo else ''}.json"
        for path in (out_dir / name, out_dir / ("latest_demo.json" if demo else "latest.json")):
            json.dump(payload, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return payload

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="판정일 D (기본: 오늘, 한국시간)")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    out = run(a.date, demo=a.demo)
    for r in out["장소"]:
        line = f"{r['장소']:<10} {r['단계']:<6} {r.get('이유', '')}"
        if r.get("예측"):
            line += "  | " + ", ".join(f"{v['날짜'][5:]} {v['하위10']:,}~{v['상위90']:,}" for v in r["예측"].values())
        print(line)
