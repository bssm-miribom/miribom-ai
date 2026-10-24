"""기준 모델: 키워드 규칙. 학습 모델이 이것보다 F1이 높아야 쓸 이유가 있음

  python3 -m miribom_ai.intent.baseline data/intent_labels.csv
"""
import re, sys
from sklearn.metrics import f1_score, precision_score, recall_score
from miribom_ai.intent.data import load, split

KEYWORDS = ["가자", "갈래", "가볼", "가봐야", "갈까", "가야지", "가려고", "다녀옴", "다녀왔", "갔다왔", "보고왔", "보고 왔", "주말에", "내일"]
_RE = re.compile("|".join(KEYWORDS))

def predict(texts):
    return [int(bool(_RE.search(t))) for t in texts]

def report(y, p, name):
    print(f"{name}: F1 {f1_score(y, p):.3f} / 정밀도 {precision_score(y, p, zero_division=0):.3f}"
          f" / 재현율 {recall_score(y, p):.3f} ({len(y)}건)")

if __name__ == "__main__":
    _, _, te = split(load(sys.argv[1] if len(sys.argv) > 1 else "data/intent_labels.csv"))
    report(te["label"].tolist(), predict(te["text"]), "키워드 규칙 (평가)")
