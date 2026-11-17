"""매일 판정 전에 어제 Threads 글을 모아 장소별 '가볼래 글 수'를 data/intent_counts.csv에 더함

  python3 -m miribom_ai.intent.daily [--date 2026-09-19]

- 서버 06시 작업이 run_daily 전에 부름 (THREADS_ACCESS_TOKEN, 의도 모델 필요)
- 글 원문·작성자는 저장하지 않고 날짜·장소별 건수만 남김
- 같은 날짜·장소가 이미 있으면 새 값으로 바꿈
"""
import argparse, os, time
import pandas as pd, requests
from miribom_ai.paths import ROOT

API = "https://graph.threads.com/v1.0"
OUT = ROOT / "data" / "intent_counts.csv"
MAX_PAGES = 5


class ThreadsError(RuntimeError):
    pass


def _search(token, q, since, until):
    rows, params = [], dict(q=q, search_type="RECENT", fields="id,text,timestamp", limit=100,
                            since=since, until=until, access_token=token)
    for _ in range(MAX_PAGES):
        r = requests.get(f"{API}/keyword_search", params=params, timeout=30)
        if r.status_code != 200:
            raise ThreadsError(f"Threads 검색 실패 {r.status_code}: {r.text[:200]}")
        res = r.json()
        rows += [d for d in res.get("data", []) if d.get("text")]
        after = res.get("paging", {}).get("cursors", {}).get("after")
        if not after or after == params.get("after"):
            break
        params["after"] = after
        time.sleep(0.5)
    return rows


def collect(day, token, places=None):
    """day(자료 기준일) 하루치 글 → 표(text, place, created_at). 글은 메모리에만 둠"""
    places = places if places is not None else pd.read_csv(ROOT / "places.csv")
    d0 = pd.Timestamp(day).tz_localize("Asia/Seoul")
    since, until = int(d0.timestamp()), int((d0 + pd.Timedelta(days=1)).timestamp())
    rows = []
    for _, p in places.iterrows():
        seen = set()
        for kw in str(p["keywords"]).split("|")[:5]:      # 장소당 검색 5회까지 (24시간 2,200회 한도)
            for d in _search(token, kw, since, until):
                if d["id"] not in seen:
                    seen.add(d["id"])
                    rows.append({"text": d["text"], "place": p["place"], "created_at": d["timestamp"]})
    return pd.DataFrame(rows, columns=["text", "place", "created_at"])


def save_counts(counts):
    counts = counts.copy()
    counts["date"] = counts["date"].astype(str)
    if OUT.exists():
        old = pd.read_csv(OUT, dtype={"date": str})
        key = set(zip(counts["date"], counts["place"]))
        old = old[[(d, p) not in key for d, p in zip(old["date"], old["place"])]]
        counts = pd.concat([old, counts], ignore_index=True)
    counts.sort_values(["date", "place"]).to_csv(OUT, index=False, encoding="utf-8-sig")


def run(day=None, token=None):
    """어제(기본) 글을 모아 분류하고 건수를 저장. 반환: 저장한 줄 수"""
    from miribom_ai.intent.predict import daily_counts
    token = token or os.environ.get("THREADS_ACCESS_TOKEN")
    if not token:
        raise ThreadsError("THREADS_ACCESS_TOKEN 없음")
    day = pd.Timestamp(day) if day else pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None) - pd.Timedelta(days=1)
    places = pd.read_csv(ROOT / "places.csv")
    posts = collect(day, token, places)
    if posts.empty:
        counts = pd.DataFrame({"date": day.date(), "place": places["place"], "intent_posts": 0, "all_posts": 0})
    else:
        counts = daily_counts(posts)
        # 글이 하나도 없던 장소도 0으로 남겨야 평소 대비 배수가 맞음
        miss = set(places["place"]) - set(counts["place"])
        counts = pd.concat([counts, pd.DataFrame({"date": day.date(), "place": sorted(miss),
                                                  "intent_posts": 0, "all_posts": 0})], ignore_index=True)
    save_counts(counts)
    return len(counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    a = ap.parse_args()
    print(f"저장 {run(a.date)}줄 → {OUT}")
