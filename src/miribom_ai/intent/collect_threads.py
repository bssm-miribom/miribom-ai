"""Threads 키워드 검색으로 라벨링할 글을 모음

사용:
  python3 -m miribom_ai.intent.collect_threads --check        토큰·권한 확인만 함
  python3 -m miribom_ai.intent.collect_threads --exchange     1시간 토큰을 60일 토큰으로 바꿔 .env에 저장함 (THREADS_APP_SECRET 필요)
  python3 -m miribom_ai.intent.collect_threads [--since 2026-08-01] [--per-place 200]

- .env에 THREADS_ACCESS_TOKEN을 넣을 것 (Threads 앱 토큰. 페이스북 Graph 토큰과 다름)
- 검색어는 places.csv의 keywords를 씀. 글쓴이 정보(username)는 받지 않음
- 결과: data/intent_labels.csv에 label이 빈 줄로 추가됨 (이미 있는 post_id는 건너뜀). 이 파일은 깃허브에 안 올라감
- 앱 심사(threads_keyword_search) 전에는 내 글만 검색됨. 한도는 24시간 2,200회
"""
import argparse, os, sys, time
import pandas as pd, requests
from miribom_ai.paths import ROOT

API = "https://graph.threads.com/v1.0"
FIELDS = "id,text,timestamp,media_type,is_reply,is_quote_post"
MAX_PAGES = 10
OUT = ROOT / "data" / "intent_labels.csv"
COLS = ["post_id", "text", "label", "place", "created_at", "labeler"]

def _get(path, token, **params):
    r = requests.get(f"{API}/{path}", params={**params, "access_token": token}, timeout=30)
    if r.status_code != 200:
        err = r.json().get("error", {}) if r.headers.get("content-type", "").startswith("application/json") else {}
        sys.exit(f"API 오류 {r.status_code}: {err.get('message', r.text[:200])}")
    return r.json()

def check(token):
    me = _get("me", token, fields="id")
    print(f"토큰 정상 (Threads 사용자 id {me['id']})")
    res = _get("keyword_search", token, q="부산", search_type="RECENT", fields="id", limit=25)
    n = len(res.get("data", []))
    print(f"'부산' 최근 글 검색 결과 {n}건")
    if n == 0:
        print("0건이면 아직 threads_keyword_search 승인 전이라 내 글만 검색되는 상태일 가능성이 큼")

def exchange(token):
    """생성기에서 받은 단기 토큰(1시간) → 장기 토큰(60일). .env의 THREADS_ACCESS_TOKEN 줄을 바꿈"""
    secret = os.environ.get("THREADS_APP_SECRET")
    if not secret:
        sys.exit(".env에 THREADS_APP_SECRET(Threads 앱 시크릿)을 넣을 것")
    r = requests.get("https://graph.threads.net/access_token", timeout=30, params={
        "grant_type": "th_exchange_token", "client_secret": secret, "access_token": token})
    if r.status_code != 200:
        sys.exit(f"교환 실패 {r.status_code}: {r.text[:200]}")
    new, days = r.json()["access_token"], r.json().get("expires_in", 0) // 86400
    env = ROOT / ".env"
    lines = [l for l in env.read_text(encoding="utf-8").splitlines() if not l.startswith("THREADS_ACCESS_TOKEN=")]
    env.write_text("\n".join(lines + [f"THREADS_ACCESS_TOKEN={new}"]) + "\n", encoding="utf-8")
    print(f"장기 토큰으로 바꿔 .env에 저장함 (약 {days}일 유효)")

def search(token, q, since, per_place):
    rows, ids, params = [], set(), dict(q=q, search_type="RECENT", fields=FIELDS, limit=100, since=since)
    for _ in range(MAX_PAGES):          # 같은 페이지를 계속 돌려주는 경우가 있어 쪽수를 제한함
        res = _get("keyword_search", token, **params)
        fresh = [d for d in res.get("data", []) if d.get("text") and d["id"] not in ids]
        rows += fresh
        ids.update(d["id"] for d in fresh)
        after = res.get("paging", {}).get("cursors", {}).get("after")
        if not fresh or not after or after == params.get("after") or len(rows) >= per_place:
            break
        params["after"] = after
        time.sleep(1)
    return rows[:per_place]

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--exchange", action="store_true")
    ap.add_argument("--since", default=(pd.Timestamp.today() - pd.Timedelta(days=90)).strftime("%Y-%m-%d"))
    ap.add_argument("--per-place", type=int, default=200)
    a = ap.parse_args(argv)
    token = os.environ.get("THREADS_ACCESS_TOKEN")
    if not token:
        sys.exit(".env에 THREADS_ACCESS_TOKEN을 넣을 것")
    if a.exchange:
        return exchange(token)
    if a.check:
        return check(token)
    places = pd.read_csv(ROOT / "places.csv")
    old = pd.read_csv(OUT, dtype={"post_id": str}) if OUT.exists() else pd.DataFrame(columns=COLS)
    seen, new = set(old["post_id"].astype(str)), []
    for _, p in places.iterrows():
        got = 0
        for kw in str(p["keywords"]).split("|"):
            for d in search(token, kw, a.since, a.per_place):
                if d["id"] in seen:
                    continue
                seen.add(d["id"]); got += 1
                new.append({"post_id": d["id"], "text": d["text"], "label": "", "place": p["place"],
                            "created_at": d["timestamp"], "labeler": ""})
        print(f"{p['place']}: 새 글 {got}건", flush=True)
    df = pd.concat([old, pd.DataFrame(new, columns=COLS)], ignore_index=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} (전체 {len(df)}건, 라벨 안 된 글 {int((df['label'].astype(str).isin(['', 'nan'])).sum())}건)")

if __name__ == "__main__":
    main()
