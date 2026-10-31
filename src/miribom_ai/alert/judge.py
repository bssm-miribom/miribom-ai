"""경보 판정 (규칙): 예측 방문객 → 병목 구간 밀집도(명/㎡) → 1·2·3단계, 판정 보류

밀집도 = 하루 방문객 × 피크 1시간 비중 × 병목 통과 비율 ÷ 병목 면적
  피크 1시간 비중: 도시철도 시간대별 승하차로 구함 [팀 확인]
  병목 통과 비율: 현장 확인 전에는 추정치 [팀 확인]
"""
import math

LIMIT = 5.0   # 1㎡당 5명 [팀 확인: 출처 표기]

def density(visitors, place):
    return visitors * place["peak_share"] * place["bottleneck_share"] / place["bottleneck_m2"]

def judge(pred, place):
    """pred: {h: {하위10, 중앙값, 상위90}} → 가장 높은 단계와 이유"""
    need = ("bottleneck_m2", "peak_share", "bottleneck_share")
    missing = [k for k in need if k not in place or place[k] is None or (isinstance(place[k], float) and math.isnan(place[k]))]
    if missing:
        return {"단계": "판정 보류", "이유": "병목 구간 정보 없음: " + ", ".join(missing), "밀집도": None}
    best, why, dens = 1, "예측 범위 상단도 1㎡당 5명 이하", {}
    for h, p in pred.items():
        mid, hi = density(p["중앙값"], place), density(p["상위90"], place)
        dens[p["날짜"]] = {"중앙값": round(mid, 2), "상위90": round(hi, 2)}
        if p.get("최악") is not None:   # 최악 시나리오 밀집도: 참고로만 보여주고 단계에는 안 씀
            dens[p["날짜"]]["최악"] = round(density(p["최악"], place), 2)
        if mid > LIMIT and best < 3:
            best, why = 3, f"{p['날짜']} 예측 중앙값 밀집도 {mid:.1f}명/㎡ > {LIMIT:g}"
        elif hi > LIMIT and best < 2:
            best, why = 2, f"{p['날짜']} 예측 범위 상단 밀집도 {hi:.1f}명/㎡ > {LIMIT:g}"
    worst = [(d, v["최악"]) for d, v in dens.items() if v.get("최악") is not None and v["최악"] > LIMIT]
    out = {"단계": f"{best}단계", "이유": why, "밀집도": dens}
    if worst:   # 단계는 그대로 두고 참고 문장만 붙임
        d, v = max(worst, key=lambda x: x[1])
        out["최악 시나리오"] = f"검색이 늘어난 만큼 방문도 늘면 {d} 밀집도 {v:.1f}명/㎡ (실측으로 확인 안 된 가정)"
    return out
