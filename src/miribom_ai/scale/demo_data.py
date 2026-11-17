"""학습 연습용 가짜 데이터 (실제 값 아님)"""
import numpy as np, pandas as pd

def make_demo(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2026-09-30")
    hol = set(pd.to_datetime(["2024-02-09", "2024-02-10", "2024-02-12", "2024-05-06", "2024-09-16", "2024-09-17",
                              "2024-09-18", "2024-10-03", "2024-10-09", "2025-01-28", "2025-01-29", "2025-01-30",
                              "2025-05-05", "2025-10-03", "2025-10-06", "2025-10-07", "2025-10-08", "2026-02-16",
                              "2026-02-17", "2026-02-18", "2026-05-05", "2026-09-24", "2026-09-25", "2026-09-26"]))
    rows = []
    for k in range(16):
        base = rng.uniform(5_000, 80_000); beach = k < 3
        mult = np.ones(len(dates)); sidx = np.ones(len(dates)) * rng.uniform(20, 40)
        for _ in range(6):                                   # 화제 사례: 검색이 먼저 뛰고 방문이 1~2일 뒤 따라옴
            s = rng.integers(60, len(dates) - 20); size = rng.uniform(1.5, 6); real = rng.random() < 0.7
            for j in range(12):
                bump = size * np.exp(-((j - 4) / 3) ** 2)
                sidx[s + j] *= 1 + bump
                if real and s + j + 2 < len(dates): mult[s + j + 2] *= 1 + 0.6 * bump
        for i, d in enumerate(dates):
            h = int(d in hol); wk = d.dayofweek >= 5
            summer = 1 + (1.5 if beach and d.month in (7, 8) else 0)
            v = base * (1.3 if wk else 1) * (1.6 if h else 1) * summer * mult[i]
            rows.append(dict(date=d, area=f"지역{k:02d}", visitors=max(0, v * rng.lognormal(0, 0.12)),
                             search_index=sidx[i] * (1.2 if wk else 1) * summer * rng.lognormal(0, 0.1),
                             holiday=h))
    return pd.DataFrame(rows)

