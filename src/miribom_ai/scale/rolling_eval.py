"""여러 시점 평가 (rolling-origin backtest): 기준 시점을 한 달씩 옮기며 매번 다시 학습하고 다음 한 달을 맞힘

시점 t마다 (정답 날짜 기준으로 나눔)
  학습: 정답 날짜 < t - 60일        검증: t - 60일 ≤ 정답 < t (조기 종료 + 범위 보정 계수)
  평가: t ≤ 정답 < 다음 달 1일       부캉이 기간(--exclude)은 학습·검증에서 뺌, 평가에서는 따로 표시
- 운영과 같은 특징(features), 같은 LightGBM 설정(PARAMS), 같은 범위 보정(CQR)을 씀
- 신뢰구간: 날짜 단위 부트스트랩 (같은 날 여러 장소는 함께 뽑음, 1,000번) 95%
- 기준 모델: 전날 값, 요일·휴일 평균 (학습 기간 평균)

  python3 -m miribom_ai.scale.rolling_eval                 # 2025-01 ~ 2026-09 매달 (21번 학습)
  python3 -m miribom_ai.scale.rolling_eval --start 2026-01 # 시점 수 줄이기
"""
import argparse, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from miribom_ai.scale.config import QUANTILES, HORIZONS, PARAMS
from miribom_ai.scale.features import build, features, add_weather
from miribom_ai.paths import ROOT

VAL_DAYS = 60
COVERAGE = 0.8
N_BOOT = 1000

def boot_ci(df, fn, seed=0):
    """날짜 단위 부트스트랩 95% 구간"""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("tgt")]
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(fn(pd.concat([groups[i] for i in pick])))
    return np.percentile(vals, [2.5, 97.5])

def wape(d, col="mid"):
    return float(np.sum(np.abs(d["y"] - d[col])) / np.sum(d["y"]))

def cover(d):
    return float(np.mean((d["y"] >= d["lo"]) & (d["y"] <= d["hi"])))

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(ROOT / "data" / "place_table.csv"))
    ap.add_argument("--start", default="2025-01"); ap.add_argument("--end", default="2026-09")
    ap.add_argument("--exclude", default="2026-09-10:2026-10-05")
    ap.add_argument("--weather", help="날씨 CSV (collect_weather.py 결과). 주면 강수량·최고기온을 입력에 넣음")
    ap.add_argument("--tag", default="", help="결과 파일 이름 뒤에 붙일 말 (예: _weather)")
    a = ap.parse_args(argv)
    raw = pd.read_csv(a.table)
    if a.weather:
        raw = add_weather(raw, a.weather)
    df = build(raw)
    ex0, ex1 = map(pd.Timestamp, a.exclude.split(":"))
    origins = pd.date_range(a.start, a.end, freq="MS")
    preds = []
    for t in origins:
        t_end = t + pd.offsets.MonthBegin(1)
        for h in HORIZONS:
            d = df.copy()
            d["log_base"] = np.log(d[f"base_h{h}"])
            d["y_log"] = np.log(d[f"y_vis_h{h}"] / d[f"base_h{h}"])
            d = d[np.isfinite(d["y_log"]) & np.isfinite(d["log_base"])]
            d["tgt"] = d["date"] + pd.Timedelta(days=h + 1)
            in_ex = (d["tgt"] >= ex0) & (d["tgt"] <= ex1)
            tr = d[(d["tgt"] < t - pd.Timedelta(days=VAL_DAYS)) & ~in_ex]
            va = d[(d["tgt"] >= t - pd.Timedelta(days=VAL_DAYS)) & (d["tgt"] < t) & ~in_ex]
            te = d[(d["tgt"] >= t) & (d["tgt"] < t_end)]
            if te.empty or len(tr) < 500: continue
            feats = features(h, d.columns)
            pv, pt = {}, {}
            for q in QUANTILES:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
                m.fit(tr[feats], tr["y_log"], eval_set=[(va[feats], va["y_log"])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
                pv[q], pt[q] = m.predict(va[feats]), m.predict(te[feats])
            vl, vh = np.minimum(pv[0.1], pv[0.5]), np.maximum(pv[0.9], pv[0.5])
            s = np.maximum(vl - va["y_log"].to_numpy(), va["y_log"].to_numpy() - vh)
            c = max(0.0, float(np.quantile(s, min(1.0, math.ceil((len(s) + 1) * COVERAGE) / len(s)))))
            base = te[f"base_h{h}"].to_numpy()
            key = ["area", f"dow_h{h}", f"hol_h{h}"]
            avg = tr.groupby(key)[f"y_vis_h{h}"].mean().rename("avg")
            seas = te.join(avg, on=key)["avg"].fillna(te[f"base_h{h}"]).to_numpy()
            lo0, hi0 = np.minimum(pt[0.1], pt[0.5]), np.maximum(pt[0.9], pt[0.5])
            preds.append(pd.DataFrame({
                "origin": t.strftime("%Y-%m"), "h": h, "area": te["area"].to_numpy(), "tgt": te["tgt"].to_numpy(),
                "y": te[f"y_vis_h{h}"].to_numpy(), "mid": base * np.exp(pt[0.5]),
                "lo": base * np.exp(lo0 - c), "hi": base * np.exp(hi0 + c),
                "lo_raw": base * np.exp(lo0), "hi_raw": base * np.exp(hi0), "c": c,
                "prev": te["prev_visitors"].to_numpy(), "seas": seas,
                "surge": te[f"y_vis_h{h}"].to_numpy() / base >= 2,
                "bukangi": ((te["tgt"] >= ex0) & (te["tgt"] <= ex1)).to_numpy()}))
        print(f"{t:%Y-%m} 완료", flush=True)
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(ROOT / "results" / f"rolling_preds{a.tag}.csv", index=False, encoding="utf-8-sig")

    rows = []
    main_set = P[~P["bukangi"]]
    for h in HORIZONS:
        for part, sub in (("전체(부캉이 제외)", main_set[main_set["h"] == h]),
                          ("급증일(실측 2배 이상)", main_set[(main_set["h"] == h) & main_set["surge"]]),
                          ("부캉이 기간", P[(P["h"] == h) & P["bukangi"]])):
            if sub.empty: continue
            r = {"예측": f"D+{h}", "구간": part, "일수(장소×날)": len(sub), "날짜 수": sub["tgt"].nunique(),
                 "WAPE": wape(sub), "WAPE 95%": boot_ci(sub, wape) if sub["tgt"].nunique() > 5 else None,
                 "범위적중률": cover(sub), "적중률 95%": boot_ci(sub, cover) if sub["tgt"].nunique() > 5 else None,
                 "보정 전 적중률": float(np.mean((sub["y"] >= sub["lo_raw"]) & (sub["y"] <= sub["hi_raw"]))),
                 "기준 전날 WAPE": wape(sub, "prev"), "기준 요일·휴일 WAPE": wape(sub, "seas")}
            for k in ("WAPE 95%", "적중률 95%"):
                r[k] = None if r[k] is None else f"{r[k][0]:.3f}~{r[k][1]:.3f}"
            rows.append(r)
    res = pd.DataFrame(rows)
    by_fold = (main_set.groupby(["origin", "h"])
               .apply(lambda g: pd.Series({"WAPE": wape(g), "범위적중률": cover(g), "c": g["c"].iloc[0]}))
               .reset_index())
    pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(res.to_string(index=False))
    print("\n시점별 D+1:"); print(by_fold[by_fold["h"] == 1].to_string(index=False))
    res.to_csv(ROOT / "results" / f"rolling_eval{a.tag}.csv", index=False, encoding="utf-8-sig")
    by_fold.to_csv(ROOT / "results" / f"rolling_eval_by_origin{a.tag}.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
