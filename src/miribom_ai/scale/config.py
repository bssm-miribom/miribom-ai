"""규모 예측 설정값 (시작값, 검증 기간 성능으로 조정)"""
QUANTILES = [0.1, 0.5, 0.9]
HORIZONS = [1, 2, 3]          # 판정일 D 기준 D+1~3. 자료는 D-1까지라 기준 행에서 h+1일 뒤를 맞힘
VIS_LAG = 28                  # 방문객 실측은 공개가 늦으므로 28일 넘게 지난 값만 평소 값 계산에 씀 [팀 확인]
BASE_DAYS = 84                # 평소 방문객 계산 기간 (28~84일 전)
PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
INTEREST = ["ratio", "ratio_max3", "ratio_growth3", "intent_posts", "intent_growth3"]
