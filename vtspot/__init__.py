"""vtspot -- video text spotting trained from scratch.

Detection, tracking and recognition in one model, with no pretrained weights of
any kind.  See docs/RESEARCH_REVIEW.md for the design rationale.
"""

__version__ = "0.1.0"

__all__ = [
    "VideoTextSpotter", "SpotterConfig", "build_model",
    "VideoTextPredictor", "PredictConfig",
    "VideoTextTracker", "TrackerConfig", "Detection",
    "Charset", "evaluate", "evaluate_dataset",
]


def __getattr__(name):
    # Lazy so that `import vtspot` stays cheap and does not pull in torch for
    # code that only needs the dataset schema or the metrics.
    if name in ("VideoTextSpotter", "SpotterConfig", "build_model"):
        from .models import spotter
        return getattr(spotter, name)
    if name in ("VideoTextPredictor", "PredictConfig"):
        from . import predictor
        return getattr(predictor, name)
    if name in ("VideoTextTracker", "TrackerConfig", "Detection"):
        from .tracking import tracker
        return getattr(tracker, name)
    if name == "Charset":
        from .utils.charset import Charset
        return Charset
    if name in ("evaluate", "evaluate_dataset"):
        from .eval import metrics
        return getattr(metrics, name)
    raise AttributeError(f"module 'vtspot' has no attribute {name!r}")
