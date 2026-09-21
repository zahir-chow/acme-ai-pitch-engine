"""Unit tests for platform reporting client and error decoupling (Part 4)."""

import pytest
import requests
from unittest.mock import MagicMock, patch
from pitch_engine.config.schema import PlatformReportingConfig
from pitch_engine.core.exceptions import PlatformReportingError
from pitch_engine.reporting.client import PlatformReportingClient
from pitch_engine.reporting.schemas import JobEventPayload, JobProgressPayload


def test_payload_validation():
    progress = JobProgressPayload(
        job_id="job_123",
        progress_percent=55.5,
        current_frame=150,
        total_frames=300,
        elapsed_seconds=5.2,
        valid_detections=10,
        status="processing",
    )
    assert progress.progress_percent == 55.5

    event = JobEventPayload(
        job_id="job_123",
        event_type="job_started",
        payload={"sample_fps": 2.0}
    )
    assert event.event_type == "job_started"
    assert event.timestamp is not None


def test_reporting_client_success():
    config = PlatformReportingConfig(
        enabled=True,
        api_base_url="http://mock-service:5000",
        max_retries=0
    )
    client = PlatformReportingClient(config)

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
        progress = JobProgressPayload(
            job_id="job_abc",
            progress_percent=100.0,
            current_frame=600,
            total_frames=600,
            elapsed_seconds=10.0,
            valid_detections=35,
        )
        success = client.report_progress(progress)
        assert success is True
        assert mock_post.called


def test_reporting_network_failure_is_decoupled():
    # Crucial Requirement: A network failure to mock_api must NOT crash the pipeline
    config = PlatformReportingConfig(
        enabled=True,
        api_base_url="http://unreachable-host:9999",
        timeout_seconds=0.1,
        max_retries=0,
        fail_on_reporting_error=False  # Decoupled
    )
    client = PlatformReportingClient(config)

    with patch.object(client.session, "post", side_effect=requests.ConnectionError("Connection refused")):
        progress = JobProgressPayload(
            job_id="job_abc",
            progress_percent=10.0,
            current_frame=60,
            total_frames=600,
            elapsed_seconds=1.0,
            valid_detections=3,
        )
        # Must return False and log warning, without raising an exception!
        success = client.report_progress(progress)
        assert success is False


def test_reporting_network_failure_raises_when_configured():
    config = PlatformReportingConfig(
        enabled=True,
        api_base_url="http://unreachable-host:9999",
        timeout_seconds=0.1,
        max_retries=0,
        fail_on_reporting_error=True  # Strictly coupled
    )
    client = PlatformReportingClient(config)

    with patch.object(client.session, "post", side_effect=requests.ConnectionError("Connection refused")):
        event = JobEventPayload(job_id="job_abc", event_type="job_failed")
        with pytest.raises(PlatformReportingError):
            client.report_event(event)
