"""평가 지표: MAE, WAPE, 범위 적중률, 범위 폭"""
import numpy as np

def metrics(y, lo, mid, hi):
    r = {"MAE": float(np.mean(np.abs(y - mid))), "WAPE": float(np.sum(np.abs(y - mid)) / np.sum(y))}
    if lo is not None:
        r["범위적중률"] = float(np.mean((y >= lo) & (y <= hi)))
        r["범위폭(배)"] = float(np.median(hi / np.maximum(lo, 1)))
    return r

