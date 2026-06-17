import numpy as np
import pytest

from online_stats import OnlineStats


def random_points(rng, n_points, dim):
    return rng.normal(size=(n_points, dim))


def test_scalar_one_at_a_time_matches_numpy():
    rng = np.random.default_rng(0)
    values = rng.normal(size=50)

    stats = OnlineStats()
    for value in values:
        stats.update(value)

    assert stats.count == 50
    assert stats.mean[0] == pytest.approx(values.mean())
    assert stats.variance[0] == pytest.approx(values.var(ddof=1))


def test_scalar_batch_matches_numpy():
    rng = np.random.default_rng(1)
    values = rng.normal(size=50)

    stats = OnlineStats()
    stats.batch_update(values)

    assert stats.count == 50
    assert stats.mean[0] == pytest.approx(values.mean())
    assert stats.variance[0] == pytest.approx(values.var(ddof=1))


def test_update_1d_array_is_a_single_point():
    stats = OnlineStats()
    stats.update([1.0, 2.0])

    assert stats.dim == 2
    assert stats.count == 1
    np.testing.assert_allclose(stats.mean, [1.0, 2.0])


def test_update_rejects_2d_array():
    stats = OnlineStats()
    with pytest.raises(ValueError):
        stats.update([[1.0, 2.0], [3.0, 4.0]])


def test_init_with_batch_of_data():
    rng = np.random.default_rng(9)
    points = random_points(rng, 40, dim=3)

    stats = OnlineStats(points)

    assert stats.dim == 3
    assert stats.count == 40
    np.testing.assert_allclose(stats.mean, points.mean(axis=0))
    np.testing.assert_allclose(stats.cov, np.cov(points, rowvar=False), atol=1e-10)


def test_init_with_batch_then_more_updates():
    rng = np.random.default_rng(10)
    points = random_points(rng, 50, dim=2)

    stats = OnlineStats(points[:30])
    stats.batch_update(points[30:])

    np.testing.assert_allclose(stats.mean, points.mean(axis=0))
    np.testing.assert_allclose(stats.cov, np.cov(points, rowvar=False), atol=1e-10)


def test_multivariate_one_at_a_time_matches_numpy():
    rng = np.random.default_rng(2)
    points = random_points(rng, 100, dim=3)

    stats = OnlineStats(dim=3)
    for point in points:
        stats.update(point)

    np.testing.assert_allclose(stats.mean, points.mean(axis=0))
    np.testing.assert_allclose(stats.cov, np.cov(points, rowvar=False), atol=1e-10)


def test_multivariate_batch_matches_numpy():
    rng = np.random.default_rng(3)
    points = random_points(rng, 100, dim=4)

    stats = OnlineStats()
    stats.batch_update(points)

    np.testing.assert_allclose(stats.mean, points.mean(axis=0))
    np.testing.assert_allclose(stats.cov, np.cov(points, rowvar=False), atol=1e-10)


def test_mixed_single_points_and_batches():
    rng = np.random.default_rng(4)
    points = random_points(rng, 30, dim=2)

    stats = OnlineStats()
    stats.batch_update(points[:10])
    for point in points[10:20]:
        stats.update(point)
    stats.batch_update(points[20:])

    np.testing.assert_allclose(stats.mean, points.mean(axis=0))
    np.testing.assert_allclose(stats.cov, np.cov(points, rowvar=False), atol=1e-10)


def test_combine_two_batches_with_plus():
    rng = np.random.default_rng(5)
    points = random_points(rng, 80, dim=3)
    split = 33

    a = OnlineStats()
    a.batch_update(points[:split])
    b = OnlineStats()
    b.batch_update(points[split:])

    combined = a + b

    assert combined.count == 80
    np.testing.assert_allclose(combined.mean, points.mean(axis=0))
    np.testing.assert_allclose(
        combined.cov, np.cov(points, rowvar=False), atol=1e-10
    )
    # originals are untouched
    assert a.count == split
    assert b.count == 80 - split


def test_iadd_combines_in_place():
    rng = np.random.default_rng(6)
    points = random_points(rng, 60, dim=2)
    split = 25

    a = OnlineStats()
    a.batch_update(points[:split])
    b = OnlineStats()
    b.batch_update(points[split:])

    a += b

    assert a.count == 60
    np.testing.assert_allclose(a.mean, points.mean(axis=0))
    np.testing.assert_allclose(a.cov, np.cov(points, rowvar=False), atol=1e-10)


def test_combine_with_empty_accumulator():
    rng = np.random.default_rng(7)
    points = random_points(rng, 20, dim=2)

    a = OnlineStats()
    a.batch_update(points)
    empty = OnlineStats()

    combined = a + empty
    np.testing.assert_allclose(combined.mean, points.mean(axis=0))
    assert combined.count == a.count

    combined2 = empty + a
    np.testing.assert_allclose(combined2.mean, points.mean(axis=0))


def test_dimension_mismatch_raises():
    a = OnlineStats(dim=2)
    a.update([1.0, 2.0])
    b = OnlineStats(dim=3)
    b.update([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        a + b

    with pytest.raises(ValueError):
        a.update([1.0, 2.0, 3.0])


def test_cov_before_any_data_raises():
    stats = OnlineStats()
    with pytest.raises(RuntimeError):
        stats.cov


def test_mean_before_any_data_raises():
    stats = OnlineStats()
    with pytest.raises(RuntimeError):
        stats.mean


def test_cov_with_too_few_points_for_ddof_raises():
    stats = OnlineStats(ddof=1)
    stats.update([1.0, 2.0])

    with pytest.raises(RuntimeError):
        stats.cov

    # mean is still available with just one point
    np.testing.assert_allclose(stats.mean, [1.0, 2.0])


def test_iadd_with_single_point():
    stats = OnlineStats()
    stats += [1.0, 2.0]
    stats += [3.0, 4.0]

    assert stats.dim == 2
    assert stats.count == 2
    np.testing.assert_allclose(stats.mean, [2.0, 3.0])


def test_iadd_with_scalar():
    stats = OnlineStats()
    stats += 1.0
    stats += 3.0

    assert stats.dim == 1
    assert stats.count == 2
    np.testing.assert_allclose(stats.mean, [2.0])


def test_iadd_with_batch_array():
    rng = np.random.default_rng(11)
    points = random_points(rng, 20, dim=3)

    stats = OnlineStats()
    stats += points

    assert stats.count == 20
    np.testing.assert_allclose(stats.mean, points.mean(axis=0))
    np.testing.assert_allclose(stats.cov, np.cov(points, rowvar=False), atol=1e-10)


def test_chained_updates_combine_many_random_splits():
    rng = np.random.default_rng(8)
    points = random_points(rng, 200, dim=5)

    total = OnlineStats()
    idx = 0
    for batch_size in [17, 1, 50, 3, 1, 128]:
        chunk = OnlineStats()
        chunk.batch_update(points[idx : idx + batch_size])
        total = total + chunk
        idx += batch_size

    np.testing.assert_allclose(total.mean, points[:idx].mean(axis=0))
    np.testing.assert_allclose(
        total.cov, np.cov(points[:idx], rowvar=False), atol=1e-8
    )
