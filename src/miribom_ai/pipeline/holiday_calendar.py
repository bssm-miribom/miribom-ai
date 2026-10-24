"""공휴일 달력 (대체공휴일 포함). 키 필요 없는 holidays 패키지를 씀"""
import holidays, pandas as pd

def holiday_set(start, end):
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    return {pd.Timestamp(d) for d in holidays.KR(years=years)}
