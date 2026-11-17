"""하루치 글을 분류해서 장소별 '가볼래 글 수'만 남김. 원문과 작성자 정보는 저장하지 않음

  python3 -m miribom_ai.intent.predict 오늘글.csv     (열: text, place, created_at)
"""
import sys, json
from pathlib import Path
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from miribom_ai.intent.config import MAX_LEN, LABELS
from miribom_ai.intent.data import clean

from miribom_ai.paths import ROOT
MODEL_DIR = ROOT / "models" / "intent"

_cache = {}

def _load():
    """모델은 한 번만 불러와 메모리에 둠 (서버에서 여러 번 부를 때)"""
    if not _cache:
        _cache["tok"] = AutoTokenizer.from_pretrained(MODEL_DIR)
        _cache["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
        _cache["th"] = json.load(open(MODEL_DIR / "threshold.json"))["threshold"]
    return _cache["tok"], _cache["model"], _cache["th"]

@torch.no_grad()
def proba(texts):
    """글마다 '가볼래' 확률"""
    tok, model, _ = _load()
    out = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        out += torch.softmax(model(**enc).logits, -1)[:, 1].tolist()
    return out

def classify(texts):
    th = _load()[2]
    return [int(p >= th) for p in proba(texts)]

def label_texts(texts):
    """서버 시연용: [{"label": "가볼래"|"그냥 언급", "score": 확률}]. 입력 문장은 저장하지 않음"""
    th = _load()[2]
    return [{"label": LABELS[int(p >= th)], "score": round(p, 3)} for p in proba([clean(t) for t in texts])]

def daily_counts(posts):
    posts = posts.assign(intent=classify(posts["text"].map(clean).tolist()))
    posts["date"] = pd.to_datetime(posts["created_at"]).dt.date
    counts = posts.groupby(["date", "place"]).agg(intent_posts=("intent", "sum"), all_posts=("intent", "size"))
    return counts.reset_index()                          # 글 원문은 여기서 버려짐

if __name__ == "__main__":
    c = daily_counts(pd.read_csv(sys.argv[1]))
    out = ROOT / "data" / "intent_counts.csv"
    c.to_csv(out, mode="a", header=not out.exists(), index=False, encoding="utf-8-sig")
    print(c.to_string(index=False))
