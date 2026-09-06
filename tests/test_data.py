"""Data pipeline: synthesis fidelity, targets, batching, converters."""
import json
import numpy as np
import cv2
import pytest
import torch
from PIL import ImageFont

from vtspot.data.converters import icdar_video, json_video, roadtext
from vtspot.data.converters.common import validate_annotation
from vtspot.data.dataset import (ClipConfig, SyntheticStaticDataset,
                                 SyntheticVideoDataset, VideoClipDataset,
                                 collate_clips)
from vtspot.data.schema import VideoAnnotation, normalise_text
from vtspot.data.synth_static import (SynthConfig, SyntheticImageGenerator,
                                      discover_fonts, render_word_layer)
from vtspot.data.synth_video import SyntheticVideoGenerator, VideoSynthConfig
from vtspot.data.targets import build_db_targets, decode_db_polygons
from vtspot.data.transforms import AugConfig, ClipAugmentor, transform_points
from vtspot.utils.charset import Charset

SMALL_AUG = AugConfig(crop_size=(160, 160), short_side=160, max_long_side=256)


@pytest.fixture(scope="module")
def fonts():
    f = discover_fonts()
    if not f:
        pytest.skip("no system fonts available")
    return f


def _fixed_fonts(n: int = 8):
    """A small, stable font list for tests that assert on generator output.

    The generators call ``discover_fonts()`` when none is passed, so their output
    depends on how many fonts the machine happens to have -- 88 on one box, 345
    on another.  Every random draw downstream shifts with it, which makes any
    seed-pinned assertion pass or fail by environment rather than by code.
    (Measured: at a fixed seed, 8 of 60 simulated font collections produced a
    clip where no instance spanned all frames.)  Sorting and truncating gives
    the tests a deterministic corpus.
    """
    found = discover_fonts()
    if not found:
        pytest.skip("no system fonts available")
    return sorted(found)[:n]


@pytest.mark.parametrize("curvature", [0.0, 0.4, -0.6])
def test_rendered_polygon_covers_the_ink(fonts, curvature):
    """The single most important synthesis invariant: if the polygon does not
    match the glyphs, every detection target derived from it is wrong."""
    font = ImageFont.truetype(fonts[0], 32)
    layer, poly = render_word_layer("MARKET", font, curvature, (255, 255, 255))
    alpha = np.array(layer)[:, :, 3] > 32
    mask = np.zeros(alpha.shape, np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    covered = (alpha & (mask > 0)).sum() / max(alpha.sum(), 1)
    density = alpha.sum() / max((mask > 0).sum(), 1)
    assert covered > 0.95, "polygon misses ink"
    assert density > 0.15, "polygon is far looser than the ink -- annotations are tight"


def test_curved_rendering_produces_non_collinear_polygons(fonts):
    font = ImageFont.truetype(fonts[0], 32)
    _, straight = render_word_layer("MARKET", font, 0.0, (255, 255, 255))
    _, curved = render_word_layer("MARKET", font, 0.8, (255, 255, 255))
    k = len(straight) // 2
    assert np.std(straight[:k, 1]) < 1.0
    assert np.std(curved[:k, 1]) > 3.0


def test_static_generator_outputs_valid_in_bounds_polygons(fonts):
    gen = SyntheticImageGenerator(SynthConfig(width=320, height=320),
                                  fonts=_fixed_fonts(), seed=0)
    img, polys, texts = gen.generate()
    assert img.shape == (320, 320, 3)
    assert len(polys) == len(texts) and len(polys) > 0
    for p in polys:
        assert p.shape[0] >= 4
        assert p.min() >= 0 and p[:, 0].max() < 320 and p[:, 1].max() < 320


def test_video_generator_preserves_identity_across_frames():
    """Instances must recur across frames, or there is nothing to associate.

    The assertion is on *multi-frame* identities rather than ones spanning the
    entire clip: an instance may legitimately drift out of frame or be occluded
    before the last frame, and training samples short windows (clip_len 4) out
    of longer clips anyway.  What the contrastive loss actually needs is
    cross-frame positives, which is what this checks.
    """
    spans = []
    for seed in range(4):
        gen = SyntheticVideoGenerator(
            VideoSynthConfig(width=320, height=192, num_frames=8),
            fonts=_fixed_fonts(), seed=seed)
        frames, anns = gen.generate()
        assert len(frames) == len(anns) == 8
        appearances = {}
        for a in anns:
            for inst in a:
                appearances[inst["track_id"]] = appearances.get(inst["track_id"], 0) + 1
                assert len(inst["polygon"]) >= 4
                assert 0.0 <= inst["visible_ratio"] <= 1.0
        assert appearances, f"seed {seed}: clip has no instances at all"
        spans.append(max(appearances.values()))

    assert min(spans) >= 2, f"a clip had no instance in two frames: {spans}"
    assert max(spans) >= 6, f"no clip has a long-lived instance: {spans}"


def test_video_generator_produces_occlusion_gaps():
    """Disappear/reappear events are what the long-term matcher trains on.
    Occluder placement is random, so a single clip may legitimately have none --
    check across seeds instead of pinning one."""
    seeds_with_gaps = 0
    for seed in range(6):
        gen = SyntheticVideoGenerator(
            VideoSynthConfig(width=320, height=192, num_frames=16, num_occluders=2,
                             occluder_prob=1.0), fonts=_fixed_fonts(), seed=seed)
        _, anns = gen.generate()
        ids = [{i["track_id"] for i in a} for a in anns]
        all_ids = set().union(*ids)
        gapped = sum(1 for tid in all_ids
                     if "0" in "".join("1" if tid in s else "0" for s in ids).strip("0"))
        seeds_with_gaps += gapped > 0
    assert seeds_with_gaps >= 4, "occlusion events too rare to train re-association"


def test_db_targets_shapes_and_ignore_handling():
    polys = [np.array([[20, 20], [120, 22], [121, 50], [19, 48]], np.float32),
             np.array([[140, 20], [200, 20], [200, 48], [140, 48]], np.float32)]
    t = build_db_targets(polys, [False, True], 128, 256)
    for k in ("shrink_map", "shrink_mask", "thresh_map", "thresh_mask"):
        assert t[k].shape == (1, 128, 256)
    assert t["shrink_mask"][0, 30, 170] == 0.0        # ignore region masked out
    assert t["shrink_map"][0, 35, 70] == 1.0          # trainable region marked
    assert t["thresh_map"].min() >= 0.3 - 1e-5
    assert t["thresh_map"].max() <= 0.7 + 1e-5


def test_tiny_instances_become_ignore_not_background():
    thin = [np.array([[10, 10], [60, 10], [60, 12], [10, 12]], np.float32)]
    t = build_db_targets(thin, [False], 64, 128, min_text_size=4)
    assert t["shrink_map"].sum() == 0
    assert t["shrink_mask"][0, 11, 30] == 0.0


def test_db_target_decode_roundtrip():
    poly = np.array([[20, 20], [220, 22], [221, 44], [19, 42]], np.float32)
    t = build_db_targets([poly], [False], 128, 256)
    got, scores = decode_db_polygons(t["shrink_map"][0].astype(np.float32),
                                     box_thresh=0.5)
    assert len(got) == 1 and scores[0] > 0.9
    from vtspot.utils.polygon import poly_iou
    assert poly_iou(got[0], poly) > 0.75


def test_clip_augmentation_is_consistent_across_frames():
    aug = ClipAugmentor(AugConfig(crop_size=(160, 160), frame_jitter_px=2.0),
                        training=True)
    base, hw = aug.base_homography((240, 320))
    pts = np.array([[100., 100.]], np.float32)
    a = transform_points(pts, aug.frame_homography(base, hw))
    b = transform_points(pts, aug.frame_homography(base, hw))
    assert np.abs(a - b).max() < 10.0, "per-frame jitter must be small, not a new crop"


def test_eval_letterbox_is_stride_32():
    aug = ClipAugmentor(AugConfig(short_side=320, max_long_side=640), training=False)
    _, (h, w) = aug.base_homography((360, 640))
    assert h % 32 == 0 and w % 32 == 0


def test_text_anchored_crop_hits_text_more_often():
    centres = np.array([[50, 60], [300, 200], [480, 90]], np.float32)
    hits = {}
    for anchored in (False, True):
        aug = ClipAugmentor(AugConfig(crop_size=(160, 160)), training=True)
        n = 0
        for _ in range(120):
            H, (h, w) = aug.base_homography((360, 640), centres if anchored else None)
            tp = transform_points(centres, H)
            if ((tp[:, 0] >= 0) & (tp[:, 0] < w) & (tp[:, 1] >= 0) & (tp[:, 1] < h)).any():
                n += 1
        hits[anchored] = n
    assert hits[True] > hits[False]


def test_collate_keeps_clip_identities_disjoint():
    cs = Charset.from_preset("alnum")
    ds = SyntheticVideoDataset(cs, length=2, clip_cfg=ClipConfig(clip_len=3),
                               video_cfg=VideoSynthConfig(width=256, height=160,
                                                          num_frames=4),
                               aug_cfg=SMALL_AUG, seed=5)
    batch = collate_clips([ds[0], ds[1]])
    assert batch["images"].shape[:2] == (2, 3)
    t0 = set(batch["inst_track"][batch["clip_index"] == 0].tolist())
    t1 = set(batch["inst_track"][batch["clip_index"] == 1].tolist())
    assert not (t0 & t1), "track ids must not collide across clips"
    assert int(batch["inst_frame"].max()) < 2 * 3


def test_static_dataset_produces_single_frame_clips():
    cs = Charset.from_preset("alnum")
    ds = SyntheticStaticDataset(cs, length=1, synth_cfg=SynthConfig(width=192, height=192),
                                aug_cfg=SMALL_AUG, seed=0)
    s = ds[0]
    assert s["images"].shape[0] == 1
    assert s["shrink_map"].shape[0] == 1


def test_unencodable_transcriptions_excluded_from_recognition():
    cs = Charset.from_preset("alnum")
    ds = SyntheticStaticDataset(cs, length=1, synth_cfg=SynthConfig(width=192, height=192),
                                aug_cfg=SMALL_AUG, seed=1)
    s = ds[0]
    assert s["inst_has_text"].dtype == torch.bool
    for i in range(len(s["inst_has_text"])):
        if not s["inst_has_text"][i]:
            assert int(s["inst_lengths"][i]) == 0


def test_dont_care_convention():
    assert normalise_text("###") == ("", True)
    assert normalise_text(" SHOP ") == ("SHOP", False)
    assert normalise_text(None) == ("", True)


def test_icdar_converter(tmp_path):
    (tmp_path / "v.xml").write_text(
        '<Frames><frame ID="1">'
        '<object ID="1" Transcription="SHOP" Quality="HIGH">'
        '<Point x="10" y="10"/><Point x="60" y="12"/>'
        '<Point x="60" y="30"/><Point x="10" y="28"/></object>'
        '<object ID="2" Transcription="X" Quality="LOW">'
        '<Point x="80" y="10"/><Point x="120" y="10"/>'
        '<Point x="120" y="30"/><Point x="80" y="30"/></object>'
        '</frame></Frames>')
    ann = icdar_video.convert(tmp_path / "v.xml", "v", 640, 480)
    assert ann.frames[0].frame_idx == 0, "XML frame IDs are 1-based, disk is 0-based"
    assert [i.ignore for i in ann.frames[0].instances] == [False, True]
    assert validate_annotation(ann, min_instances=1) == []


def test_json_converter_handles_key_variants(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(
        {"1": [{"points": [10, 10, 60, 12, 60, 30, 10, 28], "ID": 1,
                "transcription": "CAFE"}]}))
    (tmp_path / "b.json").write_text(json.dumps(
        {"0": [{"polygon": [[10, 10], [60, 12], [60, 30], [10, 28]],
                "tracking_id": 7, "text": "CAFE"}]}))
    a = json_video.convert(tmp_path / "a.json", "a", 640, 480, "dstext")
    b = json_video.convert(tmp_path / "b.json", "b", 640, 480, "bovtext")
    assert a.frames[0].frame_idx == 0 and b.frames[0].frame_idx == 0
    assert a.frames[0].instances[0].track_id == 1
    assert b.frames[0].instances[0].track_id == 7


def test_non_latin_becomes_ignore(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps(
        {"0": [{"points": [10, 10, 60, 12, 60, 30, 10, 28], "ID": 1,
                "transcription": "商店", "language": "Chinese"}]}))
    ann = json_video.convert(tmp_path / "c.json", "c", 640, 480, "bovtext")
    assert ann.frames[0].instances[0].ignore is True
    kept = json_video.convert(tmp_path / "c.json", "c", 640, 480, "bovtext",
                              non_latin_as_ignore=False)
    assert kept.frames[0].instances[0].ignore is False


def test_roadtext_converter(tmp_path):
    (tmp_path / "v.csv").write_text(
        "frame,track,x1,y1,x2,y2,text,legible\n"
        "1,5,10,10,60,30,STOP,1\n1,6,80,10,120,30,,0\n")
    ann = roadtext.convert(tmp_path / "v.csv", "v", 1280, 720)
    inst = ann.frames[0].instances
    assert len(inst) == 2
    assert inst[0].text == "STOP" and inst[0].ignore is False
    assert inst[1].ignore is True


def test_validator_flags_wrong_frame_size(tmp_path):
    (tmp_path / "d.json").write_text(json.dumps(
        {"0": [{"points": [10, 10, 600, 12, 600, 30, 10, 28], "ID": 1, "text": "A"},
               {"points": [20, 20, 610, 22, 610, 40, 20, 38], "ID": 2, "text": "B"}]}))
    ann = json_video.convert(tmp_path / "d.json", "d", 32, 32, "x")
    assert any("out of bounds" in p for p in validate_annotation(ann))


def test_schema_roundtrip(tmp_path):
    (tmp_path / "annotations").mkdir()
    from vtspot.data.schema import Frame, Instance
    ann = VideoAnnotation("v", 640, 480, [
        Frame(0, [Instance([[1, 1], [9, 1], [9, 5], [1, 5]], "A", 1)])])
    ann.to_json(tmp_path / "annotations" / "v.json")
    back = VideoAnnotation.from_json(tmp_path / "annotations" / "v.json")
    assert back.stats() == ann.stats()


def _make_frame_dataset(tmp_path):
    src = tmp_path / "raw" / "frames" / "vid1"
    src.mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(src / f"{i:06d}.jpg"), np.full((120, 160, 3), 128, np.uint8))
    gt = tmp_path / "raw" / "gt"
    gt.mkdir(parents=True)
    (gt / "vid1.json").write_text(json.dumps(
        {str(i): [{"points": [10, 10, 60, 12, 60, 30, 10, 28], "ID": 1,
                   "transcription": "CAFE"}] for i in range(3)}))
    return src, gt


def _run_prepare(tmp_path, out, refuse_symlinks):
    import importlib.util
    import sys
    from pathlib import Path
    from unittest import mock

    spec = importlib.util.spec_from_file_location(
        "pd", str(Path(__file__).resolve().parents[1] / "tools" / "prepare_dataset.py"))
    pd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pd)
    _, gt = _make_frame_dataset(tmp_path)
    argv = ["prepare_dataset.py", "--dataset", "dstext",
            "--frames", str(tmp_path / "raw" / "frames"),
            "--annotations", str(gt), "--out", str(out)]
    with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
        if refuse_symlinks:
            # WinError 1314: "A required privilege is not held by the client" --
            # what Windows returns for symlink creation without Developer Mode.
            with mock.patch.object(Path, "symlink_to",
                                   side_effect=OSError(1314, "privilege not held")):
                pd.main()
        else:
            pd.main()


def _load_first_clip(root):
    cs = Charset.from_preset("alnum")
    ds = VideoClipDataset(root, cs, clip_cfg=ClipConfig(clip_len=2),
                          aug_cfg=AugConfig(crop_size=(96, 96), short_side=96,
                                            max_long_side=160))
    return ds[0]


def test_prepare_uses_symlink_when_permitted(tmp_path):
    out = tmp_path / "prepared"
    _run_prepare(tmp_path, out, refuse_symlinks=False)
    ann = VideoAnnotation.from_json(out / "annotations" / "vid1.json")
    assert (out / "frames" / "vid1").is_symlink()
    assert ann.frames_dir == ""
    assert (ann.width, ann.height) == (160, 120)
    assert float(_load_first_clip(out)["images"].abs().sum()) > 0


def test_prepare_falls_back_when_symlinks_are_refused(tmp_path):
    """Windows rejects symlink creation without admin rights or Developer Mode.

    Copying every frame of every video is not an acceptable fallback, so the
    source directory is recorded in the annotation and the loader reads from
    there instead.
    """
    out = tmp_path / "prepared"
    _run_prepare(tmp_path, out, refuse_symlinks=True)
    ann = VideoAnnotation.from_json(out / "annotations" / "vid1.json")
    assert not (out / "frames" / "vid1").is_symlink()
    assert ann.frames_dir.endswith("vid1")
    # frame size must still be detected -- it is read from the source directory
    assert (ann.width, ann.height) == (160, 120)
    assert float(_load_first_clip(out)["images"].abs().sum()) > 0


def test_font_search_skips_undefined_platform_dirs(monkeypatch):
    from vtspot.data.synth_static import _font_search_dirs
    monkeypatch.delenv("WINDIR", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert not any("Microsoft" in d for d in _font_search_dirs())
    monkeypatch.setenv("WINDIR", r"C:\Windows")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\u\AppData\Local")
    dirs = _font_search_dirs()
    assert any(d.startswith(r"C:\Windows") for d in dirs)
    assert any("Microsoft" in d for d in dirs)
