# miribom-ai

미리봄(부산 급부상 장소 인파 조기 경보)의 AI 모델 패키지. 대시보드 FastAPI 서버가 이 패키지를 불러서 씀

## 구조
```
src/miribom_ai/
  __init__.py      서버가 부르는 함수 모음 (run_daily 등)
  paths.py         데이터·모델 폴더 위치 (MIRIBOM_HOME 환경변수로 바꿀 수 있음)
  surge/      급증 탐지 (학습 없음)
    config.py      설정값 (8주, 상위 1%, 2일)
    collect.py     데이터랩 수집 (API, 웹 CSV)
    detect.py      급증 배수 계산, 1단계 판정, 과거 재현
    replay.py      과거 사례 재현 실행 (부캉이)
  scale/      규모 예측 (LightGBM 학습)
    config.py      설정값 (분위수, 예측 일수, 하이퍼파라미터)
    demo_data.py   연습용 가짜 데이터
    features.py    특징·정답 만들기 (판정일 07시 전 정보만)
    metrics.py     MAE, WAPE, 범위 적중률
    collect_visitors.py  구·군별 일별 방문자 수 수집 (공공데이터포털)
    collect_search.py    구·군별 데이터랩 검색 지수 수집
    build_table.py       두 자료 + 공휴일 → data/scale_table.csv
    train.py       학습 + 기준 모델 비교 + 모델 저장
    predict.py     학습된 모델로 D+1~3 방문객 범위 예측
  intent/     방문 의도 분류 (KcELECTRA 미세조정)
    config.py      설정값 (모델 이름, 학습률, 에폭)
    data.py        라벨링 CSV 읽기, 개인정보 지우기, 작성일 순 나누기
    baseline.py    기준 모델 (키워드 규칙)
    train.py       미세조정 학습
    predict.py     하루치 글 분류 → 장소별 가볼래 글 수만 남김
  alert/
    judge.py       예측 방문객 → 병목 밀집도 → 1·2·3단계, 판정 보류
  pipeline/
    run_daily.py   하루 판정 전체 흐름 (결과를 dict로 돌려줌)
    weather.py     D+1~3 예보 (기상청 키 있으면 기상청, 없으면 Open-Meteo)
    holiday_calendar.py  공휴일 (대체공휴일 포함)
places.csv    추적 장소, 검색어(|로 구분), 좌표, 평소 방문객, 병목 구간 값
models/       학습된 모델 (가짜 데이터 모델은 models/demo/, 커밋 안 함)
data/ results/  자료, 평가 결과
```

## 설치
```
pip3 install -e .              # 규모 예측·급증 탐지
pip3 install -e ".[intent]"    # 방문 의도 분류까지 (torch, transformers)
```

## FastAPI 서버에서 쓰기
```python
import miribom_ai

result = miribom_ai.run_daily()                 # 오늘 판정 (네이버 키 + models/ 필요)
result = miribom_ai.run_daily("2026-09-20")     # 과거 날짜 재현 (A-07)
result = miribom_ai.run_daily(demo=True)        # 키 없이 가짜 데이터
```
- 결과: `{"판정일", "생성 시각", "가짜 데이터", "장소": [{"장소", "단계", "이유", "급증", "예측", "밀집도"}, ...]}`
- 매일 06시 예약 실행(APScheduler 등), 알림 발송, 화면은 서버 쪽에서 함
- `save=False`를 주면 `data/daily/`에 JSON을 저장하지 않고 dict만 돌려줌
- 모델이 없으면 `miribom_ai.ModelMissing` 예외를 냄 (가짜 모델로 운영하지 않게)
- 환경변수: `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`(필수), `KMA_SERVICE_KEY`(선택), `MIRIBOM_HOME`(선택, 이 레포 폴더 위치)

## 학습·시험 (레포 폴더에서)
```
python3 -m miribom_ai.surge.replay --fake --plot              # 급증 탐지 연습 (키 필요 없음)
python3 -m miribom_ai.surge.replay --csv 데이터랩.csv --plot   # 웹 CSV로 부캉이 재현
python3 -m miribom_ai.scale.train --demo                      # 규모 예측 학습 연습 (가짜 데이터)
python3 -m miribom_ai.scale.train                             # data/scale_table.csv로 실제 학습 → models/
python3 -m miribom_ai.intent.train data/intent_labels.csv     # 방문 의도 분류 학습 → models/intent/
python3 -m miribom_ai.pipeline.run_daily --demo               # 하루 판정 전체 흐름 시험
```

## 규모 예측 실제 학습 순서
```
export TOUR_SERVICE_KEY=...        # 공공데이터포털 '한국관광공사_빅데이터_지역별 방문자수_GW' 인증키
export NAVER_CLIENT_ID=... NAVER_CLIENT_SECRET=...
python3 -m miribom_ai.scale.collect_visitors 2024-01-01 2026-10-20
python3 -m miribom_ai.scale.collect_search 2024-01-01 2026-10-20   # 검색어는 data/area_keywords.csv
python3 -m miribom_ai.scale.build_table
python3 -m miribom_ai.scale.train
```

## 학습 데이터 형식
- 규모 예측 `data/scale_table.csv`: date, area, visitors, search_index, holiday, (선택) intent_posts, rain_h1~3, tmax_h1~3. 자세한 설명은 `scale/train.py` 맨 위
- 방문 의도 `data/intent_labels.csv`: post_id, text, label(1 가볼래 / 0 그냥 언급 / 빈칸 보류), place, created_at, labeler
- `places.csv`의 평소 방문객·병목 값은 지금 예시임 [팀 확인]

## 지킬 것
- API 키는 코드에 적지 않음 (환경변수)
- 설정값은 모두 시작값. 검증 기간 사례로 정하고 부캉이(2026.9.)는 평가에만 씀
- 부캉이 재현 때는 북항친수공원 검색어에 '부캉이', '북항 상어'를 넣지 않음. 그때는 등록 안 된 별칭이라 넣으면 미래 정보가 섞임
- 글 원문·작성자 정보, 데이터랩 원자료는 레포에 올리지 않음

## 라이선스
- 이 레포 코드: MIT (`LICENSE`)
- 사용 모델·라이브러리: KcELECTRA (MIT, github.com/Beomi/KcELECTRA), KoBERT (Apache-2.0, 비교용), LightGBM (MIT), scikit-learn·pandas·numpy (BSD)
- 데이터: 네이버 데이터랩·Threads 등 외부 자료는 각 서비스 약관을 따름
