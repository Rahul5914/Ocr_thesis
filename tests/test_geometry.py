"""Polygon geometry: the layer every other component's correctness rests on."""
import numpy as np
import pytest

from vtspot.utils.polygon import (adaptive_unclip_distance, bbox_iou_matrix,
                                  control_points_to_polygon,
                                  expand_polygon, offset_polygon,
                                  order_polygon_for_text,
                                  order_quad_clockwise, poly_area, poly_iou,
                                  polygon_to_control_points, resample_polyline,
                                  shrink_polygon, split_polygon_edges,
                                  unclip_ratio_for_shrink)

QUAD = np.array([[10, 10], [50, 12], [52, 30], [8, 28]], np.float32)


def test_quad_ordering_is_permutation_invariant():
    for perm in ([0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 2, 1, 0]):
        assert np.allclose(order_quad_clockwise(QUAD[perm]), QUAD)


def test_ordered_quad_is_clockwise_in_image_coords():
    q = order_quad_clockwise(QUAD)
    # TL is left of TR, and TL is above BL
    assert q[0, 0] < q[1, 0]
    assert q[0, 1] < q[3, 1]


def test_shrink_then_expand_recovers_most_of_the_area():
    shrunk = shrink_polygon(QUAD, 0.4)
    assert shrunk is not None and poly_area(shrunk) < poly_area(QUAD)
    ratio = unclip_ratio_for_shrink(QUAD, 0.4)
    # Not 1 - r^2: the inverse depends on the instance's aspect ratio, which is
    # exactly why DBNet ships unclip_ratio as a tuned constant near 1.5.
    assert 1.2 < ratio < 2.2
    back = expand_polygon(shrunk, ratio)
    assert back is not None
    assert poly_iou(back, QUAD) > 0.85


def test_exact_unclip_ratio_grows_with_elongation():
    """A fixed unclip ratio cannot serve both shapes -- this is why we adapt."""
    square = np.array([[0, 0], [60, 0], [60, 55], [0, 55]], np.float32)
    thin = np.array([[0, 0], [300, 0], [300, 12], [0, 12]], np.float32)
    r_square = unclip_ratio_for_shrink(square, 0.4)
    r_thin = unclip_ratio_for_shrink(thin, 0.4)
    assert r_thin > 2.5 * r_square
    assert 1.2 < r_square < 2.0


def test_adaptive_unclip_beats_fixed_ratio_on_elongated_text():
    fixed_ious, adaptive_ious = [], []
    for w, h in [(40, 20), (100, 20), (200, 15), (300, 12), (80, 10)]:
        quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32) + 20
        shrunk = shrink_polygon(quad, 0.4)
        assert shrunk is not None
        fixed = expand_polygon(shrunk, 1.5)
        adaptive = offset_polygon(shrunk, adaptive_unclip_distance(shrunk, 0.4))
        fixed_ious.append(poly_iou(fixed, quad) if fixed is not None else 0.0)
        adaptive_ious.append(poly_iou(adaptive, quad) if adaptive is not None else 0.0)
    assert np.mean(adaptive_ious) > np.mean(fixed_ious) + 0.15
    # every adaptive recovery clears the 0.5 matching threshold; the fixed one
    # does not, which is the practical consequence
    assert min(adaptive_ious) > 0.5
    assert min(fixed_ious) < 0.5


def test_iou_bounds():
    assert poly_iou(QUAD, QUAD) == pytest.approx(1.0, abs=1e-4)
    far = QUAD + np.array([1000, 1000], np.float32)
    assert poly_iou(QUAD, far) == 0.0


def test_control_point_roundtrip():
    ctrl = polygon_to_control_points(QUAD, 6)
    assert ctrl.shape == (12, 2)
    assert poly_iou(control_points_to_polygon(ctrl), QUAD) > 0.99


def test_curved_polygon_edges_are_left_to_right():
    xs = np.linspace(0, 100, 7)
    ys = 20 + 10 * np.sin(xs / 100 * np.pi)
    poly = np.concatenate([np.stack([xs, ys], 1),
                           np.stack([xs[::-1], ys[::-1] + 15], 1)]).astype(np.float32)
    top, bottom = split_polygon_edges(poly)
    assert top[0, 0] < top[-1, 0]
    assert bottom[0, 0] < bottom[-1, 0]
    assert top[:, 1].mean() < bottom[:, 1].mean()


def test_reorder_arbitrary_contour_puts_top_edge_first():
    xs = np.linspace(0, 100, 7)
    ys = 20 + 10 * np.sin(xs / 100 * np.pi)
    poly = np.concatenate([np.stack([xs, ys], 1),
                           np.stack([xs[::-1], ys[::-1] + 15], 1)]).astype(np.float32)
    rolled = np.roll(poly, 5, axis=0)
    top, bottom = split_polygon_edges(order_polygon_for_text(rolled))
    assert top[:, 1].mean() < bottom[:, 1].mean()


def test_resample_polyline_uniform():
    line = np.array([[0, 0], [10, 0], [10, 10]], np.float32)
    out = resample_polyline(line, 5)
    assert out.shape == (5, 2)
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert seg.std() < 1e-3


def test_bbox_iou_matrix_shapes_and_values():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]], np.float32)
    m = bbox_iou_matrix(a, b)
    assert m.shape == (1, 3)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(1 / 3, abs=1e-3)
    assert m[0, 2] == 0.0
    assert bbox_iou_matrix(np.zeros((0, 4), np.float32), b).shape == (0, 3)
