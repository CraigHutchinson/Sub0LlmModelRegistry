"""Sub0LlmModelRegistry: a versioned, manifest-described folder of model weights (spec/LAYOUT.md), with provider plugins."""
from .store import Registry, defaultRoot, roleOf, sha256  # noqa: F401

__version__ = "0.1.0"
