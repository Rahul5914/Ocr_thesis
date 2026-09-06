"""ICDAR 2013 / 2015 Video (Robust Reading Challenge 3) -> unified schema.

Ships one XML per video::

    <Frames>
      <frame ID="1">
        <object ID="3" Transcription="SHOP" Quality="HIGH" Language="English">
          <Point x="123" y="45"/> ... (4 points)
        </object>
      </frame>
    </Frames>

Two conventions that must be respected or the metrics are wrong:

* ``Transcription`` of ``##DONT#CARE##`` marks an ignore region.
* ``Quality="LOW"`` instances are unreadable and are excluded from the
  end-to-end task; they are converted to ignore rather than dropped, so they
  suppress false positives instead of creating them.
* Frame IDs are **1-based** in the XML and 0-based on disk after extraction.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..schema import VideoAnnotation
from .common import build_annotation, is_dont_care, parse_points


def parse_icdar_xml(xml_path: str | Path, low_quality_as_ignore: bool = True
                    ) -> Dict[int, List[dict]]:
    root = ET.parse(str(xml_path)).getroot()
    per_frame: Dict[int, List[dict]] = {}
    for frame in root.iter("frame"):
        try:
            frame_id = int(frame.get("ID", "0")) - 1        # XML is 1-based
        except ValueError:
            continue
        items: List[dict] = []
        for obj in frame.iter("object"):
            pts = [(float(p.get("x", 0)), float(p.get("y", 0)))
                   for p in obj.iter("Point")]
            if len(pts) < 4:
                continue
            text = obj.get("Transcription", "")
            quality = (obj.get("Quality", "") or "").upper()
            ignore = is_dont_care(text) or (low_quality_as_ignore and quality == "LOW")
            try:
                track_id = int(obj.get("ID", "-1"))
            except ValueError:
                track_id = -1
            items.append({"points": np.asarray(pts, np.float32),
                          "text": "" if ignore else text,
                          "track_id": track_id, "ignore": ignore})
        per_frame[frame_id] = items
    return per_frame


def convert(xml_path: str | Path, video_id: str, width: int, height: int,
            fps: float = 30.0, source: str = "icdar_video",
            low_quality_as_ignore: bool = True) -> VideoAnnotation:
    per_frame = parse_icdar_xml(xml_path, low_quality_as_ignore)
    return build_annotation(video_id, width, height, source, per_frame, fps)
