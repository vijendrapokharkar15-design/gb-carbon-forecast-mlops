"""HTTP helper with retries and exponential backoff."""
import logging
import time

import requests

logger = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


def get_json_with_retry(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    retries: int = 4,
    backoff_seconds: float = 2.0,
    timeout: int = 60,
) -> dict:
    """GET a URL and return JSON. Retries on network errors and temporary
    server errors; fails immediately on other errors (e.g. 400 bad request)."""
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url, params=params, headers={"Accept": "application/json"}, timeout=timeout
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
        else:
            if response.status_code == 200:
                return response.json()
            if response.status_code not in RETRY_STATUS:
                raise RuntimeError(
                    f"Request failed ({response.status_code}) for {response.url}: "
                    f"{response.text[:300]}"
                )
            last_error = RuntimeError(f"Temporary error {response.status_code}")

        if attempt < retries:
            wait = backoff_seconds * 2 ** (attempt - 1)
            logger.warning("Attempt %d/%d failed (%s). Retrying in %.0fs", attempt, retries, last_error, wait)
            time.sleep(wait)

    raise RuntimeError(f"Giving up after {retries} attempts: {url}") from last_error