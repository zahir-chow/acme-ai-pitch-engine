# Automated Pitch Boundary & Crop Engine

A scalable, observable, and resilient computer-vision pipeline designed to ingest broadcast match video, detect playing-field boundaries across dynamic scenes, derive optimal camera crop layouts, and report execution metrics to a central platform.

Built for the **TrackBox Software / Backend Developer Challenge**.

---

## Key Architecture & Features

### Part 1: Modular Architecture & Validated Configuration
- **Separation of Concerns**: Core domain and computer vision logic resides in the reusable `pitch_engine` library, completely decoupled from CLI argument handling and environment variables.
- **Fail-Fast Configuration**: Configuration is strictly validated at load time using **Pydantic v2** (`EngineConfig`). Invalid paths, negative frame rates, malformed aspect ratios, or out-of-range confidence scores halt execution immediately with clear diagnostics — *no silent fallbacks*.
- **Pluggable Detection Seam**: The `BaseFieldDetector` interface enables seamless swapping of field detection algorithms (e.g. `ColorThresholdDetector`, SAM segmentation, deep learning models, or sport-specific detectors like football, rugby, or soccer) without modifying the core pipeline.

### Part 2: Processing Efficiency
- **Adaptive Frame Sampling**: Instead of linearly processing every frame of a continuous 30/60 FPS stream, the engine samples at a configurable cadence (`sample_fps=2.0`), stepping through keyframes via hardware/demux seeks (`CAP_PROP_POS_FRAMES`). Compute scales with the content inspected, not raw video length.
- **Fast Scene-Cut Detection**: Lightweight $O(1)$ downsampled histogram distance checks (Bhattacharyya metric) detect broadcast camera transitions and close-ups in $<0.5$ ms.
- **Spatial Geometry Caching**: Pre-allocated canvas frame boundaries (`get_cached_frame_boundary`) and contour vertex simplification eliminate redundant memory allocations and accelerate Shapely polygon operations by $>10\times$.

### Part 3: Failure Handling & Observability
- **Structured Logging**: Built-in JSON and human-readable logging with contextual attributes (`job_id`, `frame_index`, `scene_id`, `elapsed_ms`) suited for unattended batch processing.
- **Resilient Error Boundaries**:
  - *Fatal Errors*: Missing video media, corrupt headers, or bad configurations fail fast with clear exit codes.
  - *Non-Fatal Recoverable Errors*: Transient frame decode glitches are skipped with warnings.
  - *Broadcast Noise Filtering*: Close-ups, crowd shots, and momentary detection noise are categorized (`NO_PITCH_VISIBLE`, `NOISE`, `INVALID_GEOMETRY`) and prevented from polluting spatial boundaries.
- **Temporal Boundary Aggregation**: The `BoundaryAggregator` clusters detections within continuous scenes, computing a consensus medoid boundary and deriving constrained aspect-ratio crop rectangles (`CropBox`).

### Part 4: Decoupled Platform Reporting
- **Validated Wire Payloads**: Strongly-typed Pydantic models (`JobProgressPayload`, `JobEventPayload`) dispatch heartbeat progress and lifecycle events over HTTP to `mock_api`.
- **Decoupled Network Resilience**: Network timeouts or transient outages talking to the platform API do *not* terminate the video analysis engine. Conversely, unrecoverable video engine errors notify the platform with a `job_failed` event before exiting.

---

## Directory Structure

```
acme-ai/
├── pitch_engine/                     # Reusable Core Library
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── schema.py                 # Validated Pydantic v2 configuration schema
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py             # Domain exception hierarchy
│   │   ├── geometry.py               # Cached spatial computations & polygon sanitization
│   │   └── models.py                 # Validated domain value objects & models
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py                   # FieldDetector abstract seam
│   │   ├── color_threshold.py        # High-performance color thresholding detector
│   │   ├── factory.py                # Detector registry & factory
│   │   └── mock.py                   # Deterministic test detector
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── aggregator.py             # Temporal boundary aggregation & noise rejection
│   │   ├── engine.py                 # Main PitchBoundaryEngine pipeline orchestrator
│   │   └── sampler.py                # Adaptive frame sampler & scene cut detector
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── client.py                 # Resilient HTTP reporting client (mock_api)
│   │   └── schemas.py                # Over-the-wire Pydantic models
│   └── logging_utils.py              # Structured logging utilities
├── mock_api/                         # Mock reporting service (Flask)
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── tests/                            # Comprehensive automated test suite
│   ├── test_config.py
│   ├── test_detectors.py
│   ├── test_efficiency.py
│   ├── test_aggregator.py
│   ├── test_reporting.py
│   └── test_e2e.py
├── cli.py                            # Thin execution entrypoint
├── synthetic_generator.py            # Synthetic match feed generator
├── synthetic_field_prototype.py      # Backwards-compatible prototype runner
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Production container specification
├── docker-compose.yml                # Multi-service composition (mock_api + runner)
├── DECISIONS.md                      # Engineering decisions & trade-off analysis
└── README.md                         # Documentation
```

---

## Getting Started

### 1. Prerequisites
- Python 3.12+
- Virtual environment (`.venv`)

### 2. Installation
Create and activate your virtual environment, then install dependencies:

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt flask

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt flask
```

### 3. Running Automated Tests
Run the comprehensive pytest suite:

```bash
pytest -v
```

All 24 unit and integration tests validate configuration fail-fast behavior, detector seams, adaptive sampling efficiency, noise filtering, network failure decoupling, and end-to-end execution.

---

## Usage

### Local Execution (CLI)

1. **Start the mock API reporting service** (in a separate terminal):
   ```bash
   python mock_api/app.py
   ```

2. **Run the pipeline**:
   ```bash
   python cli.py --video synthetic_pitch_feed.mp4 --sample-fps 2.0 --mock-api-url http://localhost:5000 --export-summary summary.json
   ```
   *Note: If `synthetic_pitch_feed.mp4` is not present, the CLI automatically generates the synthetic feed using `synthetic_generator.py`.*

3. **Verify API Events**:
   ```bash
   curl http://localhost:5000/api/v1/jobs/events
   ```

### Backwards-Compatible Prototype Execution
You can also run the original prototype entrypoint, which delegates cleanly to the production engine:
```bash
python synthetic_field_prototype.py
```

### Docker & Docker Compose

When Docker is available on your machine, launch the entire system (mock API and automated runner) with a single command:

```bash
docker compose up --build
```

---

## Engineering Decisions & Trade-Offs
See [DECISIONS.md](DECISIONS.md) for full documentation on:
- Operational assumptions regarding video stream stability and missing field boundaries.
- Validation strictness vs. graceful degradation policies.
- Performance trade-offs in frame sampling, histogram distance calculation, and spatial caching.
- AI/LLM pairing disclosure.
