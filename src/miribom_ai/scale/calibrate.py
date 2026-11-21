"""예측 범위 보정 (재학습 없음): 검증 기간 잔차로 하위10·상위90 폭을 넓히는 계수를 구해 meta json에 저장

방식: 분위수 회귀 보정(CQR, conformal). 로그 배수 기준으로
  점수 s = max(하위10 - 실제, 실제 - 상위90)
  c = 검증 기간 s의 80% 분위수 (유한 표본 보정 포함)
  운영: 하위10 × e^-c, 상위90 × e^c. 중앙값은 그대로
- 검증 기간 = train.py와 같은 나눔(기본 뒤에서 30~15%). 학습에는 안 쓰고 조기 종료에만 쓴 구간
- 평가 기간(뒤 15%)은 계수를 구할 때 쓰지 않고 전후 비교에만 씀
- 부캉이 기간(--exclude)은 검증에서 뺌

  python3 -m miribom_ai.scale.calibrate            # 실제 모델(models/)
  python3 -m miribom_ai.scale.calibrate --demo     # 가짜 데이터 모델(models/demo/)
"""
import argparse, json, math
import numpy as np, pandas as pd, lightgbm as lgb
from miribom_ai.scale.config import QUANTILES, HORIZONS
from miribom_ai.scale.demo_data import make_demo
from miribom_ai.scale.features import build
from miribom_ai.scale.metrics import metrics
from miribom_ai.paths import ROOT

COVERAGE = 0.8   # 하위10~상위90 범위가 맞아야 하는 비율

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--table", default=str(ROOT / "data" / "place_table.csv"))
    ap.add_argument("--val-start"); ap.add_argument("--test-start")
    ap.add_argument("--exclude", default="2026-09-10:2026-10-05")
    ap.add_argument("--dry-run", action="store_true", help="meta json에 저장하지 않고 결과만 출력")
    a = ap.parse_args(argv)

    mdir = ROOT / "models" / ("demo" if a.demo else "")
    df = build(make_demo() if a.demo else pd.read_csv(a.table))
    dates = np.sort(df["date"].unique())
    val_start = pd.Timestamp(a.val_start) if a.val_start else pd.Timestamp(dates[int(len(dates) * 0.70)])
    test_start = pd.Timestamp(a.test_start) if a.test_start else pd.Timestamp(dates[int(len(dates) * 0.85)])
    ex0, ex1 = map(pd.Timestamp, a.exclude.split(":"))
    print(f"검증 {val_start.date()}~{test_start.date()} (계수 계산) / 평가 {test_start.date()}~ (전후 비교)")

    rows = []
    for h in HORIZONS:
        meta_path = mdir / f"scale_h{h}_meta.json"
        meta = json.load(open(meta_path))
        feats = meta["features"]
        bs = {q: lgb.Booster(model_file=str(mdir / f"scale_h{h}_q{int(q*100)}.txt")) for q in QUANTILES}
        d = df.copy()
        d["log_base"] = np.log(d[f"base_h{h}"])
        d["y"] = np.log(d[f"y_vis_h{h}"] / d[f"base_h{h}"])
        d = d[np.isfinite(d["y"]) & np.isfinite(d["log_base"])]
        tgt = d["date"] + pd.Timedelta(days=h + 1)
        in_ex = (tgt >= ex0) & (tgt <= ex1)
        va = d[(d["date"] >= val_start) & (tgt < test_start) & ~in_ex]
        te = d[d["date"] >= test_start]

        def logq(part):
            x = part[feats].astype(float)
            p = {q: bs[q].predict(x) for q in QUANTILES}
            return np.minimum(p[0.1], p[0.5]), p[0.5], np.maximum(p[0.9], p[0.5])

        vl, vm, vh = logq(va)
        s = np.maximum(vl - va["y"].to_numpy(), va["y"].to_numpy() - vh)
        n = len(s)
        c = float(np.quantile(s, min(1.0, math.ceil((n + 1) * COVERAGE) / n)))
        c = max(c, 0.0)   # 범위를 좁히지는 않음

        tl, tm, th = logq(te)
        base, y = te[f"base_h{h}"].to_numpy(), te[f"y_vis_h{h}"].to_numpy()
        surge = (y / base) >= 2
        for name, cc in (("보정 전", 0.0), ("보정 후", c)):
            lo, mid, hi = base * np.exp(tl - cc), base * np.exp(tm), base * np.exp(th + cc)
            for part, mask in (("전체", np.ones_like(surge)), ("급증일", surge)):
                if mask.sum() == 0: continue
                rows.append({"예측": f"D+{h}", "": name, "구간": part, "일수": int(mask.sum()),
                             **metrics(y[mask], lo[mask], mid[mask], hi[mask])})
        print(f"D+{h}: 검증 {n}일, 보정 계수 c={c:.4f} (범위 ×e^±c = ×{math.exp(c):.3f})")
        if not a.dry_run:
            meta["interval_cal"] = {"c": round(c, 6), "coverage": COVERAGE, "method": "CQR, log ratio",
                                    "val": [str(val_start.date()), str(test_start.date())], "n_val": n}
            json.dump(meta, open(meta_path, "w"), ensure_ascii=False, indent=1)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(res.to_string(index=False))
    res.to_csv(ROOT / "results" / ("interval_cal_demo.csv" if a.demo else "interval_cal.csv"), index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
