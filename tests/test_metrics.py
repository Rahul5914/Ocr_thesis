"""Metrics: verified against cases whose values can be computed by hand."""
import pytest

from vtspot.eval.metrics import evaluate, evaluate_dataset, normalise_for_match


def box(x, y, w=40, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


GT = [{"text": "MARKET", "frames": {f: box(10 + f * 5, 20) for f in range(10)}},
      {"text": "EXIT", "frames": {f: box(200 - f * 5, 60) for f in range(10)}}]


def _copy(gt):
    return [{"text": t["text"], "frames": dict(t["frames"])} for t in gt]


def test_perfect_prediction_scores_one():
    r = evaluate(GT, _copy(GT))
    assert r.mota == pytest.approx(1.0)
    assert r.idf1 == pytest.approx(1.0)
    assert r.hota == pytest.approx(1.0)
    assert r.num_idsw == 0


def test_spotting_mode_penalises_a_misread_that_tracking_mode_ignores():
    pred = _copy(GT)
    pred[1]["text"] = "EXlT"
    tracking = evaluate(GT, pred, spotting=False)
    spotting = evaluate(GT, pred, spotting=True)
    assert tracking.mota == pytest.approx(1.0)
    assert spotting.mota == pytest.approx(0.0)     # half the instances now miss
    assert spotting.idf1 == pytest.approx(0.5)


def test_mota_is_insensitive_to_fragmentation_but_idf1_is_not():
    """The bias that motivates reporting IDF1 and HOTA alongside MOTA."""
    frag = [{"text": "MARKET", "frames": {f: box(10 + f * 5, 20) for f in range(5)}},
            {"text": "MARKET", "frames": {f: box(10 + f * 5, 20) for f in range(5, 10)}},
            {"text": "EXIT", "frames": dict(GT[1]["frames"])}]
    r = evaluate(GT, frag)
    assert r.num_idsw == 1
    assert r.mota > 0.9        # one switch out of 20 instances
    assert r.idf1 < 0.8        # but a quarter of the identity is lost
    assert r.assa < 0.8


def test_missing_detections_reduce_recall_and_mota():
    pred = [{"text": "MARKET",
             "frames": {f: box(10 + f * 5, 20) for f in range(0, 10, 2)}},
            {"text": "EXIT", "frames": dict(GT[1]["frames"])}]
    r = evaluate(GT, pred)
    assert r.recall == pytest.approx(0.75)
    assert r.mota == pytest.approx(0.75)
    assert r.num_fn == 5


def test_false_positives_reduce_precision():
    pred = _copy(GT) + [{"text": "GHOST",
                         "frames": {f: box(400, 400) for f in range(10)}}]
    r = evaluate(GT, pred)
    assert r.num_fp == 10
    assert r.precision == pytest.approx(2 / 3, abs=1e-3)
    assert r.mota == pytest.approx(0.5)


def test_hota_is_geometric_mean_of_deta_and_assa():
    pred = [{"text": "MARKET",
             "frames": {f: box(10 + f * 5, 20) for f in range(0, 10, 2)}},
            {"text": "EXIT", "frames": dict(GT[1]["frames"])}]
    r = evaluate(GT, pred)
    assert r.hota == pytest.approx((r.deta * r.assa) ** 0.5, abs=0.05)


def test_empty_prediction_scores_zero():
    r = evaluate(GT, [])
    assert r.mota == pytest.approx(0.0)
    assert r.idf1 == 0.0 and r.hota == 0.0
    assert r.num_fn == 20


def test_dataset_level_aggregation_weights_by_instances():
    small = ([{"text": "A", "frames": {0: box(0, 0)}}],
             [{"text": "A", "frames": {0: box(0, 0)}}])
    big_gt = [{"text": "B", "frames": {f: box(0, 0) for f in range(50)}}]
    big = (big_gt, [])
    r = evaluate_dataset([small, big])
    # 1 correct instance vs 50 missed -> MOTA must be dominated by the big clip
    assert r.mota < 0.05
    assert r.num_gt == 51


def test_alphabet_filtering_in_spotting_mode():
    assert normalise_for_match("Ma-rk!et", "abcdefghijklmnopqrstuvwxyz") == "market"
    pred = _copy(GT)
    pred[0]["text"] = "market"          # case differs only
    r = evaluate(GT, pred, spotting=True)
    assert r.mota == pytest.approx(1.0)


def test_illegible_ground_truth_is_not_scored_in_spotting_mode():
    gt = [{"text": "", "frames": {f: box(10, 20) for f in range(5)}}]
    pred = [{"text": "ANYTHING", "frames": {f: box(10, 20) for f in range(5)}}]
    r = evaluate(gt, pred, spotting=True)
    assert r.num_tp == 0
