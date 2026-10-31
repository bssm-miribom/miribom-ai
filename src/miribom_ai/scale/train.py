"""규모 예측 모델 학습: LightGBM 분위수 회귀로 D+1~3 하루 방문객의 하위 10%·중앙값·상위 90%를 예측함

입력 표 (data/scale_table.csv, 한 줄 = 지역 1곳의 하루)
  date          날짜 (YYYY-MM-DD)
  area          구·군 또는 장소 이름
  visitors      그날 실제 방문객 수 (관광데이터랩 등)
  search_index  그날 데이터랩 검색 지수 (지역 검색어)
  holiday       공휴일·대체휴일이면 1, 아니면 0 (주말은 자동 처리)
  intent_posts  (선택) 그날 가볼래 글 수
  rain_h1~h3, tmax_h1~h3  (선택) 판정일 07시 전에 나온 D+1~3 예보. 관측값을 넣으면 누수

사용 예
  python3 -m miribom_ai.scale.train --demo          # 가짜 데이터로 학습 연습
  python3 -m miribom_ai.scale.train                 # data/scale_table.csv로 학습
"""
import argparse, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from miribom_ai.scale.config import QUANTILES, HORIZONS, PARAMS, SURGE_ELASTICITY
from miribom_ai.scale.demo_data import make_demo
from miribom_ai.scale.features import build, features
from miribom_ai.scale.metrics import metrics
from miribom_ai.scale.predict import surge_mult

from miribom_ai.paths import ROOT

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--table", default=str(ROOT / "data" / "scale_table.csv"))
    ap.add_argument("--val-start", help="검증 기간 시작일 (기본: 뒤에서 30%% 지점)")
    ap.add_argument("--test-start", help="평가 기간 시작일 (기본: 뒤에서 15%% 지점)")
    ap.add_argument("--exclude", default="2026-09-10:2026-10-05", help="학습·검증에서 뺄 평가 전용 기간 (부캉이)")
    a = ap.parse_args(argv)

    raw = make_demo() if a.demo else pd.read_csv(a.table)
    mdir = ROOT / "models" / ("demo" if a.demo else "")   # 가짜 데이터 모델은 models/demo/에 따로 저장
    mdir.mkdir(parents=True, exist_ok=True)
    df = build(raw)
    flags = []
    for _, g in df.groupby("area"):
        r = g["ratio"]; thr = r.shift(2).rolling(365, min_periods=60).quantile(0.99)
        flags.append((r >= thr) & (r.shift(1) >= thr.shift(1)))
    df["flag"] = pd.concat(flags).reindex(df.index).fillna(False)
    dates = np.sort(df["date"].unique())
    val_start = pd.Timestamp(a.val_start) if a.val_start else pd.Timestamp(dates[int(len(dates) * 0.70)])
    test_start = pd.Timestamp(a.test_start) if a.test_start else pd.Timestamp(dates[int(len(dates) * 0.85)])
    ex0, ex1 = map(pd.Timestamp, a.exclude.split(":"))
    print(f"학습 ~{val_start.date()} / 검증 {val_start.date()}~ / 평가 {test_start.date()}~ / 학습 제외 {ex0.date()}~{ex1.date()}")

    rows, saved = [], {}
    for h in HORIZONS:
        d = df.copy()
        d["log_base"] = np.log(d[f"base_h{h}"])
        d["y"] = np.log(d[f"y_vis_h{h}"] / d[f"base_h{h}"])   # 평소 대비 배수(로그)를 맞힘
        d = d[np.isfinite(d["y"]) & np.isfinite(d["log_base"])]
        tgt_date = d["date"] + pd.Timedelta(days=h + 1)
        # 정답 날짜 기준으로 나눔: 학습 정답이 검증 기간에 걸치지 않게
        in_ex = (tgt_date >= ex0) & (tgt_date <= ex1)
        tr = d[(tgt_date < val_start) & ~in_ex]
        va = d[(d["date"] >= val_start) & (tgt_date < test_start) & ~in_ex]
        te = d[d["date"] >= test_start]
        base_te, y_te = te[f"base_h{h}"].to_numpy(), te[f"y_vis_h{h}"].to_numpy()
        surge = (y_te / base_te) >= 2          # 실제로 평소의 2배 넘게 온 날 = 급증일

        def fit_pred(feats):
            preds = {}
            for q in QUANTILES:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
                m.fit(tr[feats], tr["y"], eval_set=[(va[feats], va["y"])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
                preds[q] = base_te * np.exp(m.predict(te[feats]))
                saved[(h, q, len(feats))] = m
            lo, mid, hi = np.minimum(preds[0.1], preds[0.5]), preds[0.5], np.maximum(preds[0.9], preds[0.5])
            return lo, mid, hi

        full = features(h, d.columns)
        lo, mid, hi = fit_pred(full)
        cal = features(h, d.columns, with_interest=False)
        nol, nom, noh = fit_pred(cal)
        # 운영 방식: 1단계(급증) 판정된 날만 평소 수준 모델 × 검색 급증 배수(비례 가정)
        # 1단계 근사: 그날과 전날 배수가 모두 지난 1년 상위 1% 이상 (surge.detect 기준과 같은 뜻)
        flag = te["flag"].to_numpy(dtype=bool)
        sm = np.where(flag, surge_mult(te["ratio"].to_numpy()), 1.0)
        pml, pmm, pmh = nol * sm, nom * sm, noh * sm
        # 기준 모델
        prev = te["prev_visitors"].to_numpy()
        key = ["area", f"dow_h{h}", f"hol_h{h}"]
        avg = tr.groupby(key)[f"y_vis_h{h}"].mean().rename("avg")
        seas = te.join(avg, on=key)["avg"].fillna(te[f"base_h{h}"]).to_numpy()
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0)
        rf.fit(tr[full].fillna(-1), tr["y"])
        rfp = base_te * np.exp(rf.predict(te[full].fillna(-1)))

        for name, (l, m, u) in {"LightGBM (관심도 포함)": (lo, mid, hi), "LightGBM (관심도 뺌)": (nol, nom, noh),
                                "평소 모델 × 검색 배수 (운영)": (pml, pmm, pmh),
                                "랜덤포레스트": (None, rfp, None), "기준: 전날 값": (None, prev, None),
                                "기준: 요일·휴일 평균": (None, seas, None)}.items():
            for part, mask in (("전체", np.ones_like(surge)), ("급증일", surge), ("1단계 판정일", flag)):
                if mask.sum() == 0: continue
                r = metrics(y_te[mask], None if l is None else l[mask], m[mask], None if u is None else u[mask])
                rows.append({"예측": f"D+{h}", "모델": name, "구간": part, "일수": int(mask.sum()), **r})

        for q in QUANTILES:                                    # 운영용 = 평소 수준 모델 (관심도 뺌). 급증은 predict에서 곱함
            saved[(h, q, len(cal))].booster_.save_model(str(mdir / f"scale_h{h}_q{int(q*100)}.txt"))
        json.dump({"features": cal, "quantiles": QUANTILES, "surge": "multiply", "surge_elasticity": SURGE_ELASTICITY},
                  open(mdir / f"scale_h{h}_meta.json", "w"), ensure_ascii=False, indent=1)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(res.to_string(index=False))
    res.to_csv(ROOT / "results" / ("scale_metrics_demo.csv" if a.demo else "scale_metrics.csv"),
               index=False, encoding="utf-8-sig")
    print(f"\n모델: {mdir.relative_to(ROOT)}/scale_h*_q*.txt  결과표: results/")

if __name__ == "__main__":
    main()
