"""데이터랩 웹에서 받은 CSV를 data/raw/datalab/<구·군>.csv로 옮김

사용: python3 -m miribom_ai.surge.import_datalab [받은 폴더, 기본 ~/Downloads]
- 파일 앞부분에 적힌 주제어(구·군 이름 또는 검색어)로 어느 구·군인지 찾음
- 찾지 못한 파일은 건드리지 않고 목록만 보여줌
"""
import sys, shutil
from pathlib import Path
import pandas as pd
from miribom_ai.paths import ROOT

def _head(path, n=8):
    for enc in ("utf-8-sig", "cp949"):
        try:
            with open(path, encoding=enc) as f:
                return "".join(f.readline() for _ in range(n))
        except UnicodeDecodeError:
            continue
    return ""

def match_area(text, areas):
    for _, r in areas.iterrows():
        if r["area"] in text:
            return r["area"]
    for _, r in areas.iterrows():
        if any(k and k in text for k in str(r["keywords"]).split("|")):
            return r["area"]
    return None

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    src = Path(argv[0]).expanduser() if argv else Path.home() / "Downloads"
    areas = pd.read_csv(ROOT / "data" / "area_keywords.csv")
    out = ROOT / "data" / "raw" / "datalab"
    out.mkdir(parents=True, exist_ok=True)
    moved, unknown = {}, []
    for f in sorted(src.glob("*.csv"), key=lambda p: p.stat().st_mtime):
        head = _head(f)
        if "데이터랩" not in head and "검색어" not in head and "주제어" not in head:
            continue
        area = match_area(head, areas)
        if area is None:
            unknown.append(f.name)
            continue
        dst = out / f"{area}.csv"
        if f.read_bytes()[:3] != b"\xef\xbb\xbf" and _head(f) and "cp949" == _enc(f):
            dst.write_text(f.read_text(encoding="cp949"), encoding="utf-8")
        else:
            shutil.copy2(f, dst)
        moved[area] = f.name
    for a, n in moved.items():
        print(f"{a}.csv  <-  {n}")
    missing = [a for a in areas["area"] if not (out / f"{a}.csv").exists()]
    if unknown:
        print("구·군을 못 찾은 파일:", ", ".join(unknown))
    print(f"완료 {16 - len(missing)}/16" + (f"  (남은 곳: {', '.join(missing)})" if missing else ""))

def _enc(path):
    try:
        path.read_text(encoding="utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp949"

if __name__ == "__main__":
    main()
