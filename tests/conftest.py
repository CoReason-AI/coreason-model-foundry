import sys
from unittest.mock import MagicMock

# Mock heavy dependencies that might be missing in the sandbox
# to allow unit tests to run.


class MockModule(MagicMock):
    @classmethod
    def __getattr__(cls, name: str) -> MagicMock:
        return MagicMock()


# Only mock if not present
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    sys.modules["sentence_transformers"] = MockModule()

try:
    import datasets  # noqa: F401
except ImportError:
    sys.modules["datasets"] = MockModule()

try:
    import trl  # noqa: F401
except ImportError:
    sys.modules["trl"] = MockModule()

# We might also need to mock torch if it's not installed
try:
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = MockModule()
