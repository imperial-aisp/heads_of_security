"""Training tests require the optional training dependencies."""

import pytest

pytest.importorskip("datasets", reason="Training tests require the train extra")
pytest.importorskip("torchtune", reason="Training tests require the train extra")
