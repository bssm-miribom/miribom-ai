"""데이터·모델 폴더 위치와 키 불러오기

- 기본 폴더는 이 레포. 서버에서 다른 곳을 쓰려면 MIRIBOM_HOME 환경변수로 바꿈
- 레포 맨 위 .env 파일(KEY=값, 한 줄에 하나)이 있으면 환경변수로 읽어 옴. .env는 커밋하지 않음
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MIRIBOM_HOME", Path(__file__).resolve().parents[2]))

def _load_env(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))   # 이미 export한 값이 있으면 그걸 씀

_load_env(ROOT / ".env")
