#!/usr/bin/env python3
"""Upload flattened issue rows to Feishu Bitable with local checkpointing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from av_eval.feishu_export import build_bitable_records, import_key


OPEN_API = "https://open.feishu.cn/open-apis"


def post_json(url: str, payload: dict, token: str = "", attempts: int = 3) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
            if result.get("code", 0) != 0:
                raise RuntimeError(f"Feishu API error: code={result.get('code')} msg={result.get('msg')}")
            return result
        except urllib.error.HTTPError as exc:
            if attempt >= attempts or (exc.code != 429 and not 500 <= exc.code < 600):
                raise
            time.sleep(min(8, 2 ** (attempt - 1)))
    raise RuntimeError("Feishu API request exhausted retries")


def tenant_token(app_id: str, app_secret: str) -> str:
    result = post_json(
        f"{OPEN_API}/auth/v3/tenant_access_token/internal",
        {"app_id": app_id, "app_secret": app_secret},
    )
    token = str(result.get("tenant_access_token", ""))
    if not token:
        raise RuntimeError("Feishu authentication returned no tenant_access_token")
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description="上传扁平问题表到飞书多维表格")
    parser.add_argument("--csv", type=Path, default=Path("output/benchmark/feishu_import.csv"))
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/benchmark/feishu_uploaded.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    with args.csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    rows = rows[: max(0, args.limit)] if args.limit else rows
    completed: set[str] = set()
    if args.checkpoint.is_file():
        completed = set(json.loads(args.checkpoint.read_text(encoding="utf-8")))
    pending = [row for row in rows if import_key(row) not in completed]
    print(f"rows={len(rows)} pending={len(pending)} execute={args.execute}")
    if not args.execute or not pending:
        return 0

    required = ("FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_APP_TOKEN", "FEISHU_TABLE_ID")
    env = {key: os.getenv(key, "").strip() for key in required}
    missing = [key for key, value in env.items() if not value]
    if missing:
        raise SystemExit("缺少飞书环境变量：" + ",".join(missing))
    token = tenant_token(env["FEISHU_APP_ID"], env["FEISHU_APP_SECRET"])
    url = (
        f"{OPEN_API}/bitable/v1/apps/{env['FEISHU_APP_TOKEN']}/"
        f"tables/{env['FEISHU_TABLE_ID']}/records/batch_create"
    )
    for start in range(0, len(pending), 100):
        batch = pending[start : start + 100]
        post_json(url, {"records": build_bitable_records(batch)}, token=token)
        completed.update(import_key(row) for row in batch)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint.write_text(
            json.dumps(sorted(completed), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
