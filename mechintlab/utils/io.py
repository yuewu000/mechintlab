
import json, os, time
from typing import Dict, Any

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def log_jsonl(obj: Dict[str, Any], path: str):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
