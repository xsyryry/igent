"""Maintainer CLI for crawling Cambridge IELTS writing questions."""

from __future__ import annotations

import argparse
import json

from project.tools.cambridge_crawler_tool import ENTRY_URL
from project.tools.data_tool import collect_cambridge_writing_questions


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl Tongzhuo Cambridge IELTS writing questions into local SQLite.")
    parser.add_argument("--entry-url", default=ENTRY_URL)
    parser.add_argument("--max-pages", type=int, default=80)
    parser.add_argument("--task-no", type=int, choices=(1, 2), default=None)
    parser.add_argument("--cambridge-book", type=int, default=None)
    parser.add_argument("--part-no", type=int, default=None)
    parser.add_argument("--no-json", action="store_true", help="Do not write per-question JSON mirror files.")
    parser.add_argument("--no-images", action="store_true", help="Do not download prompt images.")
    parser.add_argument("--insecure", action="store_true", help="Disable SSL verification for unstable target TLS.")
    parser.add_argument("--use-local-entry", action="store_true", help="Reuse the latest saved entry HTML snapshot.")
    parser.add_argument("--no-proxy", action="store_true", help="Ignore HTTP(S)_PROXY environment variables.")
    args = parser.parse_args()

    result = collect_cambridge_writing_questions(
        entry_url=args.entry_url,
        max_pages=args.max_pages,
        task_no=args.task_no,
        cambridge_book=args.cambridge_book,
        part_no=args.part_no,
        save_json=not args.no_json,
        download_images=not args.no_images,
        verify_ssl=not args.insecure,
        use_local_entry=args.use_local_entry,
        use_env_proxy=not args.no_proxy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
