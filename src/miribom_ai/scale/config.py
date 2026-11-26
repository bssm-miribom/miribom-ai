"""규모 예측 설정값 (시작값, 검증 기간 성능으로 조정)"""
QUANTILES = [0.1, 0.5, 0.9]
HORIZONS = [1, 2, 3]          # 판정일 D 기준 D+1~3. 자료는 D-1까지라 기준 행에서 h+1일 뒤를 맞힘
VIS_LAG = 28                  # 방문객 실측은 공개가 늦으므로 28일 넘게 지난 값만 평소 값 계산에 씀 [팀 확인]
BASE_DAYS = 84                # 평소 방문객 계산 기간 (28~84일 전)
PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
INTEREST = ["ratio", "ratio_max3", "ratio_growth3", "intent_posts", "intent_growth3"]

# 급증 반영 (2단 구조): 예측 = 평소 수준 모델(요일·휴일) × 급증 배수 ** SURGE_ELASTICITY
# - 구·군 학습 자료에는 평소 2배를 넘은 날이 없어서(최대 1.8배) 트리 모델이 급증 규모를 배울 수 없음
# - 1.0 = 비례 가정: 장소 검색이 평소의 N배면 방문도 평소의 N배. 학습값이 아닌 가정임 [팀 확인]
# - 장소 단위 실측 사례(팝업·축제·촬영지 등)가 생기면 그 사례로 이 값을 추정해 바꿈 (부캉이는 평가 전용)
SURGE_ELASTICITY = 1.0
