"""미리봄 AI 모델 패키지. FastAPI 서버에서 이렇게 불러씀

    import miribom_ai
    result = miribom_ai.run_daily()                    # 오늘 판정 전체 (수집 → 급증 탐지 → 규모 예측 → 경보)
    result = miribom_ai.run_daily("2026-09-20")        # 과거 날짜 재현 (A-07)
    result = miribom_ai.run_daily(demo=True)           # 키 없이 가짜 데이터

개별 기능
    miribom_ai.surge_ratio(series), miribom_ai.judge_surge(ratio, base_date)   급증 탐지
    miribom_ai.predict_place(search, base_date, holidays, base_rest, base_work)  규모 예측
    miribom_ai.judge_alert(pred, place_dict)                                     경보 단계
    miribom_ai.count_intent(posts_df)                                            방문 의도 분류 (torch 필요)
"""
from miribom_ai.pipeline.run_daily import run as run_daily, ModelMissing
from miribom_ai.surge.collect import fetch_datalab, load_datalab_csv
from miribom_ai.surge.detect import surge_ratio, judge as judge_surge
from miribom_ai.scale.predict import predict_place
from miribom_ai.alert.judge import judge as judge_alert, LIMIT

def count_intent(posts):
    """글 표(text, place, created_at) → 날짜·장소별 가볼래 글 수. torch·transformers가 무거워서 쓸 때만 불러옴"""
    from miribom_ai.intent.predict import daily_counts
    return daily_counts(posts)

__all__ = ["run_daily", "ModelMissing", "fetch_datalab", "load_datalab_csv", "surge_ratio", "judge_surge",
           "predict_place", "judge_alert", "LIMIT", "count_intent"]
