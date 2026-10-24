"""라벨링 CSV 읽기, 개인정보 지우기, 작성일 순으로 나누기

라벨링 CSV (data/intent_labels.csv) 열: post_id, text, label(1=가볼래, 0=그냥 언급, 빈칸=보류), place, created_at, labeler
"""
import re
import pandas as pd
from miribom_ai.intent.config import SPLIT, EVAL_ONLY

_PII = [(re.compile(r"https?://\S+"), " "), (re.compile(r"@[\w.]+"), "@user"),
        (re.compile(r"\d{2,3}-\d{3,4}-\d{4}"), "<전화>")]

def clean(text):
    """링크, @멘션, 전화번호를 지움"""
    t = str(text)
    for p, r in _PII:
        t = p.sub(r, t)
    return re.sub(r"\s+", " ", t).strip()

def load(path):
    df = pd.read_csv(path)
    df = df[df["label"].isin([0, 1, "0", "1"])].copy()       # 보류(빈칸)는 뺌
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].map(clean)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.drop_duplicates("text")                           # 복사·리포스트는 한 번만
    return df.sort_values("created_at").reset_index(drop=True)

def split(df):
    """무작위로 섞지 않고 작성일 순으로 나눔. 부캉이 기간 글은 평가로만 보냄"""
    e0, e1 = map(pd.Timestamp, EVAL_ONLY)
    ev_only = df[(df["created_at"] >= e0) & (df["created_at"] <= e1)]
    rest = df.drop(ev_only.index)
    n = len(rest); a, b = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
    return rest.iloc[:a], rest.iloc[a:b], pd.concat([rest.iloc[b:], ev_only])
