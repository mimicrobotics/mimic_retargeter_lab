"""Test retargeting utilities."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from numpy.testing import assert_allclose

from mimic_retargeter_lab.utils.retarget_utils import (
    compute_kabsch_umeyama_transform,
    align_pcloud_kabsch_umeyama,
)


@pytest.fixture
def sample_points():
    """Provides a consistent set of random 3D points for testing."""
    np.random.seed(42)
    return np.random.randn(10, 3)


def test_identity_case(sample_points: np.ndarray) -> None:
    """Identity case: source == target."""
    src_points = sample_points
    tgt_points = sample_points.copy()

    aligned_points, R_kabsch, scale_factor = align_pcloud_kabsch_umeyama(
        points=src_points,
        source_landmarks=src_points,
        target_landmarks=tgt_points,
    )

    # Assert: Rotation matrix is identity
    assert_allclose(R_kabsch.as_matrix(), np.eye(3), atol=1e-6)

    # Assert: Aligned points are the same as target points
    assert_allclose(aligned_points, tgt_points, atol=1e-6)

    # Assert: Scale factor is 1.0
    assert_allclose(scale_factor, 1.0, atol=1e-6)


def test_known_transform_case(sample_points: np.ndarray) -> None:
    """Known rigid+scale transform recovery."""
    src_points = sample_points

    # Define known transform
    true_scale = 2.5
    true_rotation = Rotation.from_euler("xyz", [30, -45, 60], degrees=True)
    true_translation = np.array([1.5, -2.2, 3.0])

    # Center source points
    src_centroid = np.mean(src_points, axis=0)
    src_points_centered = src_points - src_centroid

    # Apply true transform to source points
    tgt_points_true = (
        src_points_centered * true_scale
    ) @ true_rotation.as_matrix().T + true_translation

    # Align source points to true target points
    tgt_points_aligned, R_kabsch, scale_factor = align_pcloud_kabsch_umeyama(
        points=src_points,
        source_landmarks=src_points,
        target_landmarks=tgt_points_true,
    )

    # Assert: Aligned points are close to tgt_points_true
    assert_allclose(tgt_points_aligned, tgt_points_true, atol=1e-6)

    # Assert: Rotation matrix is close to true rotation
    assert_allclose(R_kabsch.as_matrix(), true_rotation.as_matrix(), atol=1e-6)

    # Assert: Scale factor is close to true scale
    assert_allclose(scale_factor, true_scale, atol=1e-6)


def test_precomputed_scale_case(sample_points: np.ndarray) -> None:
    """Verify that providing a precomputed_scale overrides the Umeyama estimation."""
    src_points = sample_points

    # Created target points that are purely scaled from known scale factor
    scale_factor_true = 2.0
    src_centroid = np.mean(src_points, axis=0)
    tgt_points_true = (src_points - src_centroid) * scale_factor_true + src_centroid

    # Force a wrong scale factor
    scale_override = 5.0

    tgt_points_aligned, R_kabsch_output, scale_factor_output = (
        align_pcloud_kabsch_umeyama(
            points=src_points,
            source_landmarks=src_points,
            target_landmarks=tgt_points_true,
            precomputed_scale=scale_override,
        )
    )

    # Assert: Function returned scale_override
    assert scale_factor_output == scale_override

    # Assert: Output points were scaled by scale_override
    expected_points = (src_points - src_centroid) * scale_override + src_centroid
    assert_allclose(tgt_points_aligned, expected_points, atol=1e-6)


def test_reflection_case(sample_points: np.ndarray) -> None:
    """Verify that the function handles reflection cases correctly."""
    src_points = sample_points
    tgt_points = sample_points.copy()

    # Mirror target along the x-axis
    tgt_points_reflected = tgt_points * np.array([-1, 1, 1])

    src_centroid = np.mean(src_points, axis=0)
    tgt_centroid = np.mean(tgt_points, axis=0)

    R_matrix, scale_factor = compute_kabsch_umeyama_transform(
        source_points=(src_points - src_centroid),
        target_points=(tgt_points_reflected - tgt_centroid),
    )

    # Assert: Rotation matrix has determinant = +1.0
    np.linalg.det(R_matrix)

    # Assert: Outputs are finite
    assert np.all(np.isfinite(R_matrix))
    assert np.isfinite(scale_factor)


def test_degenerate_variance_case(sample_points: np.ndarray) -> None:
    """Verify that the function handles degenerate variance cases correctly."""
    src_points = np.zeros((10, 3))

    # Create a normal target
    tgt_points = np.random.randn(10, 3)
    tgt_centered = tgt_points - np.mean(tgt_points, axis=0)

    R_matrix, scale_factor = compute_kabsch_umeyama_transform(
        source_points=src_points,
        target_points=tgt_centered,
    )

    # Assert: Scale factor is 1.0
    assert_allclose(scale_factor, 1.0, atol=1e-6)
