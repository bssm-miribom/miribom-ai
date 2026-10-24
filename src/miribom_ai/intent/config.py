"""방문 의도 분류 설정값 (시작값, 검증 F1로 조정)"""
MODEL_NAME = "beomi/KcELECTRA-base-v2022"   # 비교용: "skt/kobert-base-v1"
MAX_LEN = 128
BATCH = 32
LR = 5e-5
EPOCHS = 4
THRESHOLD = 0.5              # 가볼래 확률 기준값. 학습 후 검증 F1 최대 지점으로 바꿈
SPLIT = (0.70, 0.15)         # 작성일 순 학습 70% / 검증 15% / 나머지 평가
EVAL_ONLY = ("2026-09-10", "2026-10-05")    # 부캉이 기간 글은 평가에만 씀
LABELS = {0: "그냥 언급", 1: "가볼래"}
