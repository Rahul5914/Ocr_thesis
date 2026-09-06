from . import common, icdar_video, json_video, roadtext  # noqa: F401

REGISTRY = {
    "icdar13_video": icdar_video,
    "icdar15_video": icdar_video,
    "bovtext": json_video,
    "dstext": json_video,
    "artvideo": json_video,
    "roadtext1k": roadtext,
}
