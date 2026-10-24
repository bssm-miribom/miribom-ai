"""데이터·모델 폴더 위치. 기본은 이 레포 폴더이고, 서버에서 다른 곳을 쓰려면 MIRIBOM_HOME 환경변수로 바꿈"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MIRIBOM_HOME", Path(__file__).resolve().parents[2]))
