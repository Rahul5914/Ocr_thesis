"""Tracking: matching, identity preservation, and trajectory aggregation."""
import numpy as np
import pytest

from vtspot.tracking.decode import (aggregate_trajectory_text, ctc_greedy_decode,
                                    levenshtein)
from vtspot.tracking.matcher import (LongTermMatcher, MatchConfig, ShortTermMatcher,
                                     cosine_matrix, solve_assignment)
from vtspot.tracking.tracker import Detection, TrackerConfig, VideoTextTracker

E = np.eye(3, 16, dtype=np.float32)
BOXES = np.array([[10, 10, 60, 30], [100, 10, 150, 30], [200, 10, 250, 30]], np.float32)


def test_short_term_matches_small_motion():
    m = ShortTermMatcher(MatchConfig())
    matches, ut, ud = m(BOXES, E, BOXES + 3, E.copy())
    assert sorted(matches) == [(0, 0), (1, 1), (2, 2)]
    assert not ut and not ud


def test_motion_gate_breaks_ties_between_identical_appearances():
    """Repeated words are the norm in scene text; appearance alone cannot
    separate them, so the motion prior must."""
    same = np.tile(E[:1], (3, 1))
    matches, _, _ = ShortTermMatcher(MatchConfig())(BOXES, same, BOXES + 3, same.copy())
    assert sorted(matches) == [(0, 0), (1, 1), (2, 2)]


def test_teleporting_detection_is_rejected():
    far = np.array([[600, 400, 650, 420]], np.float32)
    matches, _, unmatched = ShortTermMatcher(MatchConfig())(
        BOXES[:1], E[:1], far, E[:1])
    assert matches == [] and unmatched == [0]


def test_long_term_reassociates_after_a_gap():
    lt = LongTermMatcher(MatchConfig())
    matches, _, _ = lt(BOXES[:1], E[:1], np.array([12.0]),
                       np.array([[300, 60, 350, 80]], np.float32), E[:1])
    assert matches == [(0, 0)]


def test_long_term_rejects_different_appearance():
    lt = LongTermMatcher(MatchConfig())
    matches, _, _ = lt(BOXES[:1], E[:1], np.array([12.0]),
                       np.array([[300, 60, 350, 80]], np.float32), E[1:2])
    assert matches == []


def test_assignment_respects_cost_ceiling():
    cost = np.array([[0.1, 0.9], [0.9, 0.95]], np.float32)
    matches, ur, uc = solve_assignment(cost, max_cost=0.5)
    assert matches == [(0, 0)]
    assert ur == [1] and uc == [1]
    assert solve_assignment(np.zeros((0, 3)), 0.5) == ([], [], [0, 1, 2])


def test_cosine_matrix_handles_empty():
    assert cosine_matrix(np.zeros((0, 4)), E).shape == (0, 3)


def _run_clip(occluded_frames=(), n_frames=20):
    tracker = VideoTextTracker(TrackerConfig(min_hits=2))
    rng = np.random.RandomState(0)
    texts = ["MARKET", "EXIT", "CAFE"]
    for f in range(n_frames):
        dets = []
        for i in range(3):
            if i == 1 and f in occluded_frames:
                continue
            x = 10 + i * 120 + f * 4
            poly = np.array([[x, 20 + i * 30], [x + 60, 20 + i * 30],
                             [x + 60, 44 + i * 30], [x, 44 + i * 30]], np.float32)
            emb = E[i] + rng.randn(16).astype(np.float32) * 0.05
            emb /= np.linalg.norm(emb)
            good = f % 5 != 0
            dets.append(Detection(poly, 0.9, emb,
                                  texts[i] if good else texts[i][:-1] + "X",
                                  0.85 if good else 0.4))
        tracker.update(f, dets)
    return tracker.finalise()


def test_tracker_produces_one_trajectory_per_instance():
    res = _run_clip()
    assert len(res) == 3


def test_identity_survives_a_long_occlusion():
    res = _run_clip(occluded_frames=tuple(range(6, 15)))
    assert len(res) == 3, "occlusion fragmented an identity"
    spans = {r["track_id"]: r["end_frame"] - r["start_frame"] for r in res}
    assert max(spans.values()) >= 18


def test_trajectory_aggregation_outvotes_bad_frames():
    res = _run_clip()
    assert {r["text"] for r in res} == {"MARKET", "EXIT", "CAFE"}


def test_ctc_greedy_collapses_repeats_and_blanks():
    lp = np.log(np.full((6, 5), 1e-6))
    for t, c in enumerate([1, 1, 0, 2, 2, 3]):
        lp[t, c] = np.log(0.9)
    labels, conf = ctc_greedy_decode(lp)
    assert labels == [1, 2, 3]
    assert conf == pytest.approx(0.9, abs=1e-3)


def test_ctc_greedy_on_empty():
    assert ctc_greedy_decode(np.zeros((0, 5))) == ([], 0.0)


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0


def test_aggregation_prefers_the_consistent_reading():
    obs = [("MARKET", 0.9), ("MARKET", 0.85), ("MARKET", 0.8),
           ("MARKFT", 0.4), ("MARKE", 0.3), ("WARKET", 0.35)]
    text, conf = aggregate_trajectory_text(obs)
    assert text == "MARKET" and conf > 0.5


def test_aggregation_recovers_a_word_no_single_frame_got_right():
    obs = [("MARKFT", 0.6), ("WARKET", 0.6), ("MARKET", 0.5),
           ("MARKET", 0.55), ("MARKET", 0.5)]
    assert aggregate_trajectory_text(obs)[0] == "MARKET"


def test_aggregation_on_empty():
    assert aggregate_trajectory_text([]) == ("", 0.0)


def test_recognition_rescore_vetoes_unreadable_regions():
    """A background false positive scores well on the detector but decodes to
    nothing; the geometric mean with the recogniser's confidence suppresses it."""
    cfg = TrackerConfig(use_recognition_rescore=True, rescore_weight=0.5)
    tracker = VideoTextTracker(cfg)
    real = Detection(np.array([[0, 0], [50, 0], [50, 20], [0, 20]], np.float32),
                     0.8, E[0], "SHOP", 0.9)
    junk = Detection(np.array([[0, 0], [50, 0], [50, 20], [0, 20]], np.float32),
                     0.8, E[0], "", 0.02)
    assert tracker._combined_score(real) > tracker._combined_score(junk) * 3


def test_tracker_reset_clears_state():
    tracker = VideoTextTracker(TrackerConfig(min_hits=1))
    tracker.update(0, [Detection(np.array([[0, 0], [10, 0], [10, 5], [0, 5]], np.float32),
                                 0.9, E[0], "A", 0.9)])
    assert tracker.tracks
    tracker.reset()
    assert not tracker.tracks and not tracker.lost
