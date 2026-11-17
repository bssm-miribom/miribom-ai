"""방문 의도 분류 모델 학습: KcELECTRA를 우리 라벨링 글로 미세조정 (가볼래 / 그냥 언급)

  pip3 install torch transformers scikit-learn
  python3 -m miribom_ai.intent.train data/intent_labels.csv
맥북(M 칩)은 mps, Colab은 cuda를 자동으로 씀. 1,000건 기준 Colab T4에서 수 분 (실측 후 기록)
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from miribom_ai.intent.config import MODEL_NAME, MAX_LEN, BATCH, LR, EPOCHS
from miribom_ai.intent.data import load, split
from miribom_ai.intent import baseline

from miribom_ai.paths import ROOT
OUT = ROOT / "models" / "intent"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def loader(tok, df, shuffle):
    enc = tok(df["text"].tolist(), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    ds = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(df["label"].to_numpy()))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

@torch.no_grad()
def probs(model, dl):
    model.eval(); out = []
    for ids, mask, _ in dl:
        logits = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits
        out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)

def best_threshold(y, p):
    grid = np.arange(0.2, 0.81, 0.05)
    return float(max(grid, key=lambda t: f1_score(y, (p >= t).astype(int))))

def main(path):
    torch.manual_seed(0)
    tr, va, te = split(load(path))
    print(f"학습 {len(tr)} / 검증 {len(va)} / 평가 {len(te)}건, 가볼래 비율(학습) {tr['label'].mean():.2f}, 장치 {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    dl_tr, dl_va, dl_te = loader(tok, tr, True), loader(tok, va, False), loader(tok, te, False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * len(dl_tr) * EPOCHS), len(dl_tr) * EPOCHS)
    best = -1
    for ep in range(EPOCHS):
        model.train()
        for ids, mask, y in dl_tr:
            loss = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE), labels=y.to(DEVICE)).loss
            loss.backward(); opt.step(); sch.step(); opt.zero_grad()
        f1 = f1_score(va["label"], (probs(model, dl_va) >= 0.5).astype(int))
        print(f"에폭 {ep + 1}: 검증 F1 {f1:.3f}")
        if f1 > best:                                    # 검증 F1이 가장 좋은 에폭만 저장
            best = f1; model.save_pretrained(OUT); tok.save_pretrained(OUT)
    model = AutoModelForSequenceClassification.from_pretrained(OUT).to(DEVICE)
    th = best_threshold(va["label"].to_numpy(), probs(model, dl_va))
    p_te = (probs(model, dl_te) >= th).astype(int)
    f1 = baseline.report(te["label"].tolist(), p_te.tolist(), f"KcELECTRA (평가, 기준값 {th:.2f})")
    kw = baseline.report(te["label"].tolist(), baseline.predict(te["text"]), "키워드 규칙 (평가)")
    # 서버 /admin/models 가 읽는 값 (모델 상태 카드)
    json.dump({"threshold": th, "model": MODEL_NAME, "f1": f1, "keyword_f1": kw,
               "n_train": len(tr), "n_test": len(te), "trained_at": str(pd.Timestamp.now(tz="Asia/Seoul").date()),
               "labelers": sorted(set(tr.get("labeler", pd.Series(dtype=str)).dropna()))},
              open(OUT / "threshold.json", "w"), ensure_ascii=False)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "intent_labels.csv"))
