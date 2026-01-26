from __future__ import annotations

from pathlib import Path
import os
import sys

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.integration.config import build_config, load_env

load_env()
os.environ.setdefault("SIMILAR_TOTAL", "5")
os.environ.setdefault("SIMILAR_MAX_PER_SOURCE", "1")


@pytest.fixture(scope="session")
def integration_config():
    return build_config()


@pytest.fixture(scope="session")
def api_client():
    from api.app import app

    with TestClient(app) as client:
        yield client

