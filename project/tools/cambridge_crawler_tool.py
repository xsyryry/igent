"""Maintainer-only crawler for Cambridge IELTS writing questions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from html.parser import HTMLParser
import hashlib
import json
import re
from pathlib import Path
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from requests import Response
from requests.exceptions import SSLError

from project.db.models import get_connection, init_db


ENTRY_URL = "https://ielts.itongzhuo.com/business/ielts/student/jumpIeltsQuestionList.do?sSubjects=4&type=6&aiType=0"
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_ROOT / "raw" / "cambridge_writing"
IMAGE_DIR = DATA_ROOT / "questions" / "images"
JSON_DIR = DATA_ROOT / "questions" / "writing"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) IELTS-Agent-Maintainer/1.0",
    "Accept": "text/html,image/*,*/*;q=0.8",
}


@dataclass(slots=True)
class CambridgeWritingQuestion:
    id: str
    source_site: str
    source_url: str
    cambridge_book: int | None
    part_no: int | None
    task_no: int | None
    prompt_text: str
    image_url: str
    image_local_path: str
    module: str
    question_type: str
    crawl_time: str
    parse_status: str
    raw_snapshot_path: str


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "a" and attrs_dict.get("href"):
            self.links.append(str(attrs_dict["href"]))
        if tag == "img" and attrs_dict.get("src"):
            self.images.append(str(attrs_dict["src"]))


class _CleanTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "nav", "footer", "aside"}:
            self.skip_depth += 1
        elif self.skip_depth == 0 and tag in {"h1", "h2", "h3", "p", "li", "td", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "aside"} and self.skip_depth > 0:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0 and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self.parts)).strip()


def crawl_writing_questions(
    *,
    entry_url: str = ENTRY_URL,
    max_pages: int = 80,
    save_json: bool = True,
    download_images: bool = True,
    verify_ssl: bool = True,
    use_local_entry: bool = False,
    use_env_proxy: bool = True,
) -> dict[str, Any]:
    """Crawl Tongzhuo Cambridge writing questions into local SQLite."""

    init_db()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    visited: set[str] = set()
    queue = [entry_url]
    parsed: list[CambridgeWritingQuestion] = []
    failures: list[dict[str, str]] = []

    with requests.Session() as session:
        session.headers.update(REQUEST_HEADERS)
        session.trust_env = use_env_proxy
        session.verify = verify_ssl
        try:
            if use_local_entry:
                entry_html = _load_latest_entry_snapshot()
            else:
                entry_response = _request_get(session, entry_url, verify_ssl=verify_ssl)
                entry_response.raise_for_status()
                entry_html = entry_response.text
                _save_snapshot(entry_url, entry_html)
            parsed.extend(
                _crawl_api_questions(
                    session=session,
                    entry_url=entry_url,
                    entry_html=entry_html,
                    max_items=max_pages,
                    download_images=download_images,
                    verify_ssl=verify_ssl,
                )
            )
        except Exception as exc:
            failures.append({"url": entry_url, "error": f"api_crawl_failed: {exc}"})

        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            try:
                response = _request_get(session, url, verify_ssl=verify_ssl)
                response.raise_for_status()
                html = response.text
                snapshot_path = _save_snapshot(url, html)
                queue.extend(_discover_candidate_links(html, base_url=url, seen=visited, limit=max_pages))
                question = _parse_question_page(
                    html,
                    source_url=url,
                    raw_snapshot_path=str(snapshot_path),
                    crawl_time=datetime.utcnow().replace(microsecond=0).isoformat(),
                    session=session,
                    download_images=download_images,
                )
                if question:
                    parsed.append(question)
            except Exception as exc:
                failures.append({"url": url, "error": str(exc)})

    saved = _upsert_questions(parsed)
    if save_json:
        _write_json_records(parsed)

    return {
        "skill": "crawl_cambridge_writing_questions_skill",
        "entry_url": entry_url,
        "visited_pages": len(visited),
        "parsed_count": len(parsed),
        "saved_count": saved,
        "db_table": "writing_questions",
        "raw_dir": str(RAW_DIR),
        "image_dir": str(IMAGE_DIR),
        "json_dir": str(JSON_DIR),
        "fail_count": len(failures),
        "failures": failures[:10],
    }


def _save_snapshot(url: str, html: str) -> Path:
    stem = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    path = RAW_DIR / f"{stem}.html"
    path.write_text(html, encoding="utf-8")
    return path


def _save_json_snapshot(name: str, data: Any) -> Path:
    stem = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
    path = RAW_DIR / f"{stem}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_latest_entry_snapshot() -> str:
    snapshots = sorted(RAW_DIR.glob("*.html"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not snapshots:
        raise FileNotFoundError(f"No local entry snapshot found in {RAW_DIR}")
    return snapshots[0].read_text(encoding="utf-8", errors="ignore")


def _request_get(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    verify_ssl: bool = True,
) -> Response:
    last_error: Exception | None = None
    candidates = [url]
    if url.startswith("https://"):
        candidates.append("http://" + url[len("https://") :])
    for candidate_url in candidates:
        for attempt in range(3):
            try:
                return session.get(
                    candidate_url,
                    params=params,
                    timeout=(5, 30),
                    verify=verify_ssl,
                )
            except SSLError as exc:
                last_error = exc
                if verify_ssl:
                    try:
                        return session.get(
                            candidate_url,
                            params=params,
                            timeout=(5, 30),
                            verify=False,
                        )
                    except Exception as inner_exc:
                        last_error = inner_exc
            except Exception as exc:
                last_error = exc
            time.sleep(1 + attempt)
    raise last_error or RuntimeError(f"GET failed: {url}")


def _crawl_api_questions(
    *,
    session: requests.Session,
    entry_url: str,
    entry_html: str,
    max_items: int,
    download_images: bool,
    verify_ssl: bool,
) -> list[CambridgeWritingQuestion]:
    if max_items <= 0:
        return []
    base_url = f"{urlparse(entry_url).scheme}://{urlparse(entry_url).netloc}"
    me_type = _extract_hidden_value(entry_html, "meType") or "1"
    ai_type = _extract_hidden_value(entry_html, "aiType") or "0"
    crawl_time = datetime.utcnow().replace(microsecond=0).isoformat()
    records: list[CambridgeWritingQuestion] = []

    for task_no in (1, 2):
        list_url = urljoin(base_url, "/business/single/stu/querySingleListWriting.do")
        list_data = _get_json(
            session,
            list_url,
            params={
                "sPart": task_no,
                "meType": "5" if me_type == "3" else me_type,
                "answerQuestionType": ai_type,
            },
            verify_ssl=verify_ssl,
        )
        _save_json_snapshot(f"list_task_{task_no}_{list_url}", list_data)
        items = _extract_single_items(list_data)
        for item in items:
            if len(records) >= max_items:
                return records
            s_id = item.get("sId") or item.get("sid") or item.get("id")
            if not s_id:
                continue
            detail_url = urljoin(base_url, "/business/single/sys/getFrameList.do")
            detail_data = _get_json(session, detail_url, params={"sId": s_id}, verify_ssl=verify_ssl)
            detail_snapshot = _save_json_snapshot(f"detail_{s_id}_{detail_url}", detail_data)
            question = _parse_api_question(
                item=item,
                detail_data=detail_data,
                task_no=task_no,
                source_url=f"{detail_url}?sId={s_id}",
                raw_snapshot_path=str(detail_snapshot),
                crawl_time=crawl_time,
                session=session,
                download_images=download_images,
            )
            if question:
                records.append(question)
    return records


def _get_json(session: requests.Session, url: str, *, params: dict[str, Any], verify_ssl: bool) -> dict[str, Any]:
    response = _request_get(session, url, params=params, verify_ssl=verify_ssl)
    response.raise_for_status()
    return response.json()


def _extract_hidden_value(html: str, element_id: str) -> str:
    match = re.search(rf'id=["\']{re.escape(element_id)}["\'][^>]*value=["\']([^"\']*)', html)
    return match.group(1).strip() if match else ""


def _extract_single_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    return_data = data.get("returnData") if isinstance(data, dict) else {}
    if not isinstance(return_data, dict):
        return []
    items = return_data.get("mockExamSingleList") or return_data.get("singleList") or return_data.get("list") or []
    return items if isinstance(items, list) else []


def _parse_api_question(
    *,
    item: dict[str, Any],
    detail_data: dict[str, Any],
    task_no: int,
    source_url: str,
    raw_snapshot_path: str,
    crawl_time: str,
    session: requests.Session,
    download_images: bool,
) -> CambridgeWritingQuestion | None:
    return_data = detail_data.get("returnData") if isinstance(detail_data, dict) else {}
    if not isinstance(return_data, dict):
        return None
    single = return_data.get("single") if isinstance(return_data.get("single"), dict) else {}
    frame_list = return_data.get("frameList") if isinstance(return_data.get("frameList"), list) else []
    html_parts = _extract_topic_html_parts(frame_list)
    prompt = _extract_prompt_from_html_parts(html_parts)
    if not prompt:
        return None
    title_text = " ".join(str(value) for value in (item.get("sName"), single.get("sName"), single.get("sTitle")) if value)
    cambridge_book = _extract_cambridge_book(title_text, item, single)
    part_no = _extract_part_no(title_text, item, single)
    image_url = _extract_image_url_from_detail(detail_data, html_parts=html_parts, source_url=source_url)
    image_local_path = (
        _download_image(image_url, session=session, verify_ssl=session.verify) if image_url and download_images else ""
    )
    parse_status = "parsed"
    if _prompt_requires_image(prompt) and not image_url:
        parse_status = "partial_missing_image"
    elif image_url and download_images and not image_local_path:
        parse_status = "partial_image_download_failed"
    record_id = _question_id(cambridge_book, part_no, task_no, prompt)
    return CambridgeWritingQuestion(
        id=record_id,
        source_site="itongzhuo",
        source_url=source_url,
        cambridge_book=cambridge_book,
        part_no=part_no,
        task_no=task_no,
        prompt_text=prompt,
        image_url=image_url,
        image_local_path=image_local_path,
        module="writing",
        question_type="cambridge",
        crawl_time=crawl_time,
        parse_status=parse_status,
        raw_snapshot_path=raw_snapshot_path,
    )


def _extract_topic_html_parts(frame_list: list[Any]) -> list[str]:
    parts: list[str] = []
    for frame in frame_list:
        if not isinstance(frame, dict):
            continue
        for topic in frame.get("topicList") or []:
            if isinstance(topic, dict) and topic.get("tContent"):
                parts.append(str(topic["tContent"]))
        if frame.get("tContent"):
            parts.append(str(frame["tContent"]))
    return parts


def _extract_prompt_from_html_parts(html_parts: list[str]) -> str:
    text = "\n".join(_clean_text(part) for part in html_parts if part)
    return _extract_prompt(text)


def _extract_image_url_from_html_parts(html_parts: list[str], *, source_url: str) -> str:
    for html in html_parts:
        image_url = _extract_image_url(html, base_url=source_url)
        if image_url:
            return image_url
    return ""


def _extract_image_url_from_detail(detail_data: dict[str, Any], *, html_parts: list[str], source_url: str) -> str:
    html_image = _extract_image_url_from_html_parts(html_parts, source_url=source_url)
    if html_image:
        return html_image
    base_url = f"{urlparse(source_url).scheme}://{urlparse(source_url).netloc}"
    for value in _iter_image_candidates(detail_data):
        url = _normalize_image_url(str(value), base_url=base_url)
        if url:
            return url
    return ""


def _iter_image_candidates(value: Any) -> list[str]:
    candidates: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if isinstance(item, str) and any(token in key_lower for token in ("image", "img", "fileurl", "f_file", "ffile")):
                candidates.append(item)
            candidates.extend(_iter_image_candidates(item))
    elif isinstance(value, list):
        for item in value:
            candidates.extend(_iter_image_candidates(item))
    elif isinstance(value, str) and re.search(r"\.(png|jpe?g|webp|gif)(?:\?|$)", value, flags=re.IGNORECASE):
        candidates.append(value)
    return candidates


def _normalize_image_url(value: str, *, base_url: str) -> str:
    value = value.strip()
    if not value or not re.search(r"\.(png|jpe?g|webp|gif)(?:\?|$)", value, flags=re.IGNORECASE):
        return ""
    if value.startswith("//"):
        return "https:" + value
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("/"):
        return urljoin(base_url, value)
    return urljoin("https://ai-oss3.oss-cn-shenzhen.aliyuncs.com/exam/", value)


def _prompt_requires_image(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(
        marker in lowered or marker in prompt
        for marker in ("chart", "charts", "graph", "table", "diagram", "map", "below", "下图", "图表", "表格")
    )


def _extract_cambridge_book(text: str, item: dict[str, Any], single: dict[str, Any]) -> int | None:
    for value in (item.get("sMeType2"), item.get("meType2"), single.get("sMeType2"), single.get("meType2")):
        try:
            number = int(value)
            if 1 <= number <= 30:
                return number
        except (TypeError, ValueError):
            pass
    return _extract_int(text, (r"剑(?:桥)?雅思\s*(\d{1,2})", r"Cambridge\s*(\d{1,2})", r"雅思真题\s*(\d{1,2})", r"C\s*(\d{1,2})"))


def _extract_part_no(text: str, item: dict[str, Any], single: dict[str, Any]) -> int | None:
    for value in (item.get("sTestNum"), item.get("testNum"), single.get("sTestNum"), single.get("testNum")):
        try:
            number = int(value)
            if 1 <= number <= 10:
                return number
        except (TypeError, ValueError):
            pass
    return _extract_int(text, (r"Test\s*(\d)", r"Part\s*(\d)", r"雅思真题\d{1,2}[-_\s]*Test\s*(\d)"))


def _discover_candidate_links(html: str, *, base_url: str, seen: set[str], limit: int) -> list[str]:
    parser = _LinkExtractor()
    parser.feed(html)
    links: list[str] = []
    for href in parser.links:
        url = urljoin(base_url, href)
        if url in seen:
            continue
        if "itongzhuo.com" not in urlparse(url).netloc:
            continue
        if not _looks_like_question_url(url):
            continue
        links.append(url)
        if len(links) >= limit:
            break
    return links


def _looks_like_question_url(url: str) -> bool:
    lowered = url.lower()
    return (
        any(token in lowered for token in ("question", "ielts", "writing", "jump"))
        and ("sSubjects=4" in url or "ssubjects=4" in lowered)
    )


def _parse_question_page(
    html: str,
    *,
    source_url: str,
    raw_snapshot_path: str,
    crawl_time: str,
    session: requests.Session,
    download_images: bool,
) -> CambridgeWritingQuestion | None:
    text = _clean_text(html)
    book = _extract_int(text, (r"剑(?:桥)?雅思\s*(\d{1,2})", r"Cambridge\s*(\d{1,2})", r"剑\s*(\d{1,2})"))
    part_no = _extract_int(text, (r"Part\s*(\d)", r"part\s*(\d)", r"第\s*(\d)\s*部分"))
    task_no = _extract_int(text, (r"Task\s*(\d)", r"task\s*(\d)", r"小作文|Task\s*1", r"大作文|Task\s*2"))
    if task_no is None:
        task_no = 1 if "小作文" in text else 2 if "大作文" in text else None
    prompt = _extract_prompt(text)
    if not prompt:
        return None

    image_url = _extract_image_url(html, base_url=source_url)
    image_local_path = ""
    if image_url and download_images:
        image_local_path = _download_image(image_url, session=session, verify_ssl=session.verify)

    record_id = _question_id(book, part_no, task_no, prompt)
    return CambridgeWritingQuestion(
        id=record_id,
        source_site="itongzhuo",
        source_url=source_url,
        cambridge_book=book,
        part_no=part_no,
        task_no=task_no,
        prompt_text=prompt,
        image_url=image_url,
        image_local_path=image_local_path,
        module="writing",
        question_type="cambridge",
        crawl_time=crawl_time,
        parse_status="parsed",
        raw_snapshot_path=raw_snapshot_path,
    )


def _clean_text(html: str) -> str:
    parser = _CleanTextExtractor()
    parser.feed(html)
    text = parser.text()
    noise = ("登录", "注册", "会员", "VIP", "充值", "联系我们", "相关推荐", "评论", "分享")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(line for line in lines if not any(token in line for token in noise))


def _extract_int(text: str, patterns: tuple[str, ...]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        if match.groups():
            return int(match.group(1))
        if "小作文" in pattern:
            return 1
        if "大作文" in pattern:
            return 2
    return None


def _extract_prompt(text: str) -> str:
    patterns = (
        r"(?:作文题目|题目|Writing Task\s*\d)[:：]?\s*(.{60,1200})",
        r"((?:The graph|The chart|The table|The diagram|Some people|Nowadays|In many countries).{60,1200})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            prompt = _normalize_prompt(match.group(1))
            if _valid_prompt(prompt):
                return prompt
    return ""


def _normalize_prompt(text: str) -> str:
    stop_markers = ("参考范文", "解析", "答案", "上一题", "下一题", "评论")
    for marker in stop_markers:
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
    return re.sub(r"\s+", " ", text).strip()


def _valid_prompt(text: str) -> bool:
    if len(text) < 40:
        return False
    return any(token.lower() in text.lower() for token in ("graph", "chart", "table", "diagram", "some people", "nowadays", "write", "essay"))


def _extract_image_url(html: str, *, base_url: str) -> str:
    parser = _LinkExtractor()
    parser.feed(html)
    for src in parser.images:
        if re.search(r"\.(png|jpe?g|webp|gif)(?:\?|$)", src, flags=re.IGNORECASE):
            return urljoin(base_url, src)
    return ""


def _download_image(url: str, *, session: requests.Session, verify_ssl: bool) -> str:
    try:
        response = _request_get(session, url, verify_ssl=verify_ssl)
        response.raise_for_status()
        suffix = Path(urlparse(url).path).suffix or ".jpg"
        path = IMAGE_DIR / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()[:16]}{suffix}"
        path.write_bytes(response.content)
        return str(path)
    except Exception as exc:
        failure_path = IMAGE_DIR / "_download_failures.log"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("a", encoding="utf-8") as file:
            file.write(f"{datetime.utcnow().isoformat()} {url} {exc}\n")
        return ""


def _question_id(book: int | None, part_no: int | None, task_no: int | None, prompt: str) -> str:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"cambridge_{book or 'unknown'}_part_{part_no or 'unknown'}_task_{task_no or 'unknown'}_{digest}"


def _upsert_questions(records: list[CambridgeWritingQuestion]) -> int:
    if not records:
        return 0
    saved = 0
    with get_connection() as connection:
        for record in records:
            connection.execute(
                """
                INSERT OR IGNORE INTO writing_questions (
                    id, source_site, source_url, cambridge_book, part_no, task_no,
                    prompt_text, image_url, image_local_path, module, question_type,
                    crawl_time, parse_status, raw_snapshot_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.source_site,
                    record.source_url,
                    record.cambridge_book,
                    record.part_no,
                    record.task_no,
                    record.prompt_text,
                    record.image_url,
                    record.image_local_path,
                    record.module,
                    record.question_type,
                    record.crawl_time,
                    record.parse_status,
                    record.raw_snapshot_path,
                ),
            )
            saved += 1 if connection.execute("SELECT changes()").fetchone()[0] else 0
        connection.commit()
    return saved


def _write_json_records(records: list[CambridgeWritingQuestion]) -> None:
    for record in records:
        path = JSON_DIR / f"{record.id}.json"
        path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")
