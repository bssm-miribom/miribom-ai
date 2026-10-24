"""급증 탐지 설정값. 모두 시작값이며 검증 기간 사례로 정함 (부캉이는 평가 전용)"""
BASELINE_WEEKS = 8      # 평소 값 기간 (4·8·12주 비교)
TOP_PCT = 1.0           # 지난 1년 상위 몇 % (0.5·1·2% 비교)
CONSEC_DAYS = 2         # 연속 일수 (1·2·3일 비교)
HISTORY_DAYS = 365      # 비교할 과거 기간
FETCH_DAYS = HISTORY_DAYS + BASELINE_WEEKS * 7 + 14  # 한 번에 받아올 기간
DATALAB_URL = "https://openapi.naver.com/v1/datalab/search"  # API HUB 주소가 다르면 바꿈 [팀 확인]
