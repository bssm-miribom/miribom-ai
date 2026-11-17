"""경보 판정 (규칙): 예측 방문객 → 병목 구간 밀집도(명/㎡) → 1·2·3단계, 판정 보류

단계 (D+1~3 중 가장 높은 단계, 기준 LIMIT = 5명/㎡)
  3단계: 실측 모델 중앙값 밀집도 > 5
  2단계: 실측 모델 상위90 밀집도 > 5, 또는 최악 시나리오 밀집도 > 5 (WORST_STAGE2)
  1단계: 급증 탐지(surge.detect)에서 1단계로 올라왔지만 위에 해당 없음
  판정 보류: 병목 면적·피크 비중·병목 통과 비율 중 빠진 값이 있음

밀집도 = 하루 방문객 × 피크 1시간 비중 × 병목 통과 비율 ÷ 병목 면적
  피크 1시간 비중: 도시철도 시간대별 하차, 휴일 하루 중 가장 붐빈 1시간 비중의 상위 10% 값 (2024-01~2026-09)
  병목 통과 비율: 현장 확인 전에는 추정치 [팀 확인]
"""
import math

LIMIT = 5.0   # 1㎡당 5명 [팀 확인: 출처 표기]
# 최악 시나리오(검색이 N배면 방문도 N배)가 기준을 넘으면 2단계 (2026-11-17)
# 도시철도 하차 실측으로는 급증이 거의 안 잡혀 실측 모델만으로는 2단계가 안 나옴.
# 2025-01~2026-10 장소 5곳 되돌려 보기: 최악 > 5명/㎡는 부캉이(2026-09-19~) 때만, 다른 곳 0건
WORST_STAGE2 = True

def density(visitors, place):
    return visitors * place["peak_share"] * place["bottleneck_share"] / place["bottleneck_m2"]

def judge(pred, place):
    """pred: {h: {하위10, 중앙값, 상위90}} → 가장 높은 단계와 이유"""
    need = ("bottleneck_m2", "peak_share", "bottleneck_share")
    missing = [k for k in need if k not in place or place[k] is None or (isinstance(place[k], float) and math.isnan(place[k]))]
    if missing:
        return {"단계": "판정 보류", "이유": "병목 구간 정보 없음: " + ", ".join(missing), "밀집도": None,
                "빠진 값": missing}
    best, why, dens = 1, "예측 범위 상단도 1㎡당 5명 이하", {}
    for h, p in pred.items():
        mid, hi = density(p["중앙값"], place), density(p["상위90"], place)
        dens[p["날짜"]] = {"중앙값": round(mid, 2), "상위90": round(hi, 2)}
        if p.get("최악") is not None:   # 최악 시나리오 밀집도: 5명/㎡ 넘으면 아래에서 2단계 (3단계는 안 줌)
            dens[p["날짜"]]["최악"] = round(density(p["최악"], place), 2)
        if mid > LIMIT and best < 3:
            best, why = 3, f"{p['날짜']} 예측 중앙값 밀집도 {mid:.1f}명/㎡ > {LIMIT:g}"
        elif hi > LIMIT and best < 2:   # 실측 모델 범위 상단
            best, why = 2, f"{p['날짜']} 예측 범위 상단 밀집도 {hi:.1f}명/㎡ > {LIMIT:g}"
    worst = [(d, v["최악"]) for d, v in dens.items() if v.get("최악") is not None and v["최악"] > LIMIT]
    if worst:
        d, v = max(worst, key=lambda x: x[1])
        if WORST_STAGE2 and best < 2:
            best, why = 2, f"최악 시나리오 {d} 밀집도 {v:.1f}명/㎡ > {LIMIT:g} (검색 급증만큼 방문도 늘 경우)"
    out = {"단계": f"{best}단계", "이유": why, "밀집도": dens}
    if worst:
        out["최악 시나리오"] = f"검색이 늘어난 만큼 방문도 늘면 {d} 밀집도 {v:.1f}명/㎡ (실측으로 확인 안 된 가정)"
    return out
