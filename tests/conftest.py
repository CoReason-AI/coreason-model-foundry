import sys
from unittest.mock import MagicMock

# Mock heavy dependencies that might be missing in the sandbox
# to allow unit tests to run.


class MockModule(MagicMock):
    @classmethod
    def __getattr__(cls, name: str) -> MagicMock:
        return MagicMock()


# Only mock if not present
if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MockModule()

if "datasets" not in sys.modules:
    sys.modules["datasets"] = MockModule()

if "trl" not in sys.modules:
    sys.modules["trl"] = MockModule()

# We might also need to mock torch if it's not installed
try:
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = MockModule()
