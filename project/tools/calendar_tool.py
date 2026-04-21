"""Calendar tool with clear mock behavior and future real-service extension points."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


CALENDAR_BACKEND_ENV = "CALENDAR_BACKEND"
GOOGLE_CALENDAR_CREDENTIALS_ENV = "GOOGLE_CALENDAR_CREDENTIALS"
GOOGLE_CALENDAR_ID_ENV = "GOOGLE_CALENDAR_ID"


def _is_real_calendar_configured() -> bool:
    """Check whether a real calendar backend is configured.

    In a future Google Calendar integration, this function can be expanded to
    validate OAuth credentials, service account files, and target calendar IDs.
    """

    backend = os.getenv(CALENDAR_BACKEND_ENV, "mock_calendar").strip().lower()
    credentials = os.getenv(GOOGLE_CALENDAR_CREDENTIALS_ENV, "").strip()
    calendar_id = os.getenv(GOOGLE_CALENDAR_ID_ENV, "").strip()
    return backend == "google_calendar" and bool(credentials and calendar_id)


def create_study_event(
    title: str,
    start_time: str,
    end_time: str,
    description: str | None = None,
) -> dict[str, Any]:
    """Create a study calendar event.

    This phase keeps a clean function signature for future Google Calendar
    integration. When real calendar configuration is not available, a readable
    mock result is returned for local demos.

    Future real implementation hook:
    - Replace the placeholder branch inside `_create_google_calendar_event`
    - Keep this function signature stable so planner/tool layers do not change
    """

    if _is_real_calendar_configured():
        return _create_google_calendar_event(
            title=title,
            start_time=start_time,
            end_time=end_time,
            description=description,
        )

    logger.info("Calendar tool using mock backend for create_study_event")
    return {
        "backend": "mock_calendar",
        "status": "mock_created",
        "title": title,
        "start_time": start_time,
        "end_time": end_time,
        "description": description,
        "message": "Mock calendar event created. Configure Google Calendar env vars to enable real writes.",
    }


def get_schedule(date: str) -> list[dict[str, Any]]:
    """Fetch study schedule for a date.

    Future real implementation hook:
    - Replace the placeholder branch inside `_get_google_calendar_schedule`
    - Keep the return shape stable for downstream nodes and generator logic
    """

    if _is_real_calendar_configured():
        return _get_google_calendar_schedule(date=date)

    logger.info("Calendar tool using mock backend for get_schedule")
    return [
        {
            "backend": "mock_calendar",
            "date": date,
            "title": "IELTS Writing Practice",
            "start_time": f"{date}T09:00:00",
            "end_time": f"{date}T10:30:00",
            "duration_minutes": 90,
            "description": "Task 2 structure and idea development practice.",
            "status": "scheduled",
        },
        {
            "backend": "mock_calendar",
            "date": date,
            "title": "IELTS Reading Review",
            "start_time": f"{date}T19:00:00",
            "end_time": f"{date}T20:00:00",
            "duration_minutes": 60,
            "description": "True/False/Not Given mistake review.",
            "status": "scheduled",
        },
    ]


def _create_google_calendar_event(
    *,
    title: str,
    start_time: str,
    end_time: str,
    description: str | None,
) -> dict[str, Any]:
    """Placeholder for future Google Calendar integration.

    Suggested future implementation:
    1. Load OAuth/service-account credentials
    2. Create a Google Calendar API client
    3. Insert the event into the configured calendar
    4. Normalize the response into the same shape returned here
    """

    logger.warning("Google Calendar backend requested but real integration is not implemented yet.")
    return {
        "backend": "google_calendar",
        "status": "not_implemented",
        "title": title,
        "start_time": start_time,
        "end_time": end_time,
        "description": description,
        "message": "Google Calendar integration placeholder reached. Replace _create_google_calendar_event with a real implementation.",
    }


def _get_google_calendar_schedule(*, date: str) -> list[dict[str, Any]]:
    """Placeholder for future Google Calendar schedule retrieval."""

    logger.warning("Google Calendar backend requested but real schedule retrieval is not implemented yet.")
    return [
        {
            "backend": "google_calendar",
            "date": date,
            "title": "Calendar integration placeholder",
            "start_time": f"{date}T00:00:00",
            "end_time": f"{date}T00:30:00",
            "duration_minutes": 30,
            "description": "Replace _get_google_calendar_schedule with a real Google Calendar query.",
            "status": "not_implemented",
        }
    ]
