"""Platform reporting client with retry policy, timeout handling, and failure decoupling (Part 4)."""

import logging
import time
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pitch_engine.config.schema import PlatformReportingConfig
from pitch_engine.core.exceptions import PlatformReportingError
from pitch_engine.reporting.schemas import JobEventPayload, JobProgressPayload

logger = logging.getLogger("pitch_engine.reporting")


class PlatformReportingClient:
    """HTTP client communicating with the platform API (mock_api).

    Decouples network/reporting failures from the video processing engine.
    """

    def __init__(self, config: PlatformReportingConfig):
        self.config = config
        self.enabled = config.enabled
        self.base_url = config.api_base_url.rstrip("/")
        self.progress_url = f"{self.base_url}{config.progress_endpoint}"
        self.events_url = f"{self.base_url}{config.events_endpoint}"
        self.timeout = config.timeout_seconds
        self.fail_on_error = config.fail_on_reporting_error

        # Setup resilient requests session with retries for transient connection errors
        self.session = requests.Session()
        if config.max_retries > 0:
            retries = Retry(
                total=config.max_retries,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504],
                raise_on_status=False
            )
            adapter = HTTPAdapter(max_retries=retries)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

    def report_progress(self, payload: JobProgressPayload) -> bool:
        """Sends progress update to the orchestrator. Returns True if successful."""
        if not self.enabled:
            return True

        return self._post(self.progress_url, payload.model_dump())

    def report_event(self, payload: JobEventPayload) -> bool:
        """Sends a lifecycle or domain event to the orchestrator. Returns True if successful."""
        if not self.enabled:
            return True

        return self._post(self.events_url, payload.model_dump())

    def _post(self, url: str, data: dict) -> bool:
        try:
            resp = self.session.post(url, json=data, timeout=self.timeout)
            if resp.status_code in (200, 201, 202):
                return True

            err_msg = f"Platform service returned HTTP {resp.status_code} for {url}: {resp.text}"
            if self.fail_on_error:
                raise PlatformReportingError(err_msg, details={"status_code": resp.status_code, "url": url})
            else:
                logger.warning(f"Reporting warning: {err_msg}")
                return False

        except requests.RequestException as e:
            err_msg = f"Failed to reach platform service at {url}: {e}"
            if self.fail_on_error:
                raise PlatformReportingError(err_msg, details={"url": url, "error": str(e)}) from e
            else:
                logger.warning(f"Reporting network warning: {err_msg}")
                return False

    def close(self) -> None:
        """Closes the underlying HTTP session."""
        self.session.close()
