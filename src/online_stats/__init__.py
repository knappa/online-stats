"""Online mean and covariance statistics via the Welford/Chan algorithm.

References:
    Knuth, D. E. *The Art of Computer Programming, Volume 2:
    Seminumerical Algorithms*, section 4.2.2.
    Chan, T. F. et al. "Updating Formulae and a Pairwise Algorithm for
    Computing Sample Variances." (1979).
"""

from typing import Optional

import numpy as np

__all__ = ["OnlineStats"]


class OnlineStats:
    """Accumulates mean and covariance from a stream of data points.

    Data can be added one point at a time with `update`, or in
    batches with `batch_update`:

        stats = OnlineStats()
        stats.update([1.0, 2.0])                        # a single point
        stats.batch_update([[3.0, 4.0], [5.0, 6.0]])     # a batch of points

    `update` takes one point: a scalar, or a 1-D array-like of length
    `dim`. `batch_update` takes many points: a 1-D array-like of
    shape ``(n_points,)`` is a batch of scalar points (`dim` 1), and a
    2-D array-like of shape ``(n_points, dim)`` is a batch of vector
    points.

    Two accumulators built from disjoint batches of data can be merged
    with `+` (or `+=`), giving the same result as if all the data had
    been fed to a single accumulator, using the parallel combination
    formula of Chan et al.

    `+=` also accepts a scalar or array-like directly, instead of an
    `OnlineStats`: a 0-D or 1-D array-like is passed to `update`
    (a single point), and a 2-D array-like is passed to `batch_update`
    (a batch of points).

    Args:
        data: Optional initial batch of data points, with the same
            shape conventions as `batch_update`: a 1-D array-like of
            shape ``(n_points,)`` (scalar points, `dim` 1), or a 2-D
            array-like of shape ``(n_points, dim)``.
        dim: Dimension of each data point. If `None` (the default),
            the dimension is inferred from the first point seen.
        ddof: Delta degrees of freedom used when reporting the
            covariance matrix; the divisor is ``count - ddof``.
    """

    def __init__(self, data=None, dim=None, ddof=1):
        if dim is not None and dim < 1:
            raise ValueError("dim must be a positive integer")
        self.dim: Optional[int] = None
        self.ddof = ddof
        self.count = 0
        self._mean: Optional[np.ndarray] = None
        self._sum_sq_dev: Optional[np.ndarray] = None  # M2: sum of outer(delta, delta)
        if dim is not None:
            self._init_dim(dim)
        if data is not None:
            self.batch_update(data)

    def _init_dim(self, dim):
        self.dim = dim
        self._mean = np.zeros(dim)
        self._sum_sq_dev = np.zeros((dim, dim))

    def _add_point(self, point):
        point = np.asarray(point, dtype=float).reshape(-1)
        if self.dim is None:
            self._init_dim(point.shape[0])
        elif point.shape[0] != self.dim:
            raise ValueError(
                f"expected a point of dimension {self.dim}, got {point.shape[0]}"
            )
        assert self._mean is not None and self._sum_sq_dev is not None
        self.count += 1
        delta = point - self._mean
        self._mean += delta / self.count
        delta2 = point - self._mean
        self._sum_sq_dev += np.outer(delta, delta2)

    def _merge_batch(self, points):
        n_batch, dim = points.shape
        if self.dim is None:
            self._init_dim(dim)
        elif dim != self.dim:
            raise ValueError(
                f"expected points of dimension {self.dim}, got {dim}"
            )
        if n_batch == 0:
            return
        batch_mean = points.mean(axis=0)
        centered = points - batch_mean
        batch_sum_sq_dev = centered.T @ centered
        self._merge_stats(n_batch, batch_mean, batch_sum_sq_dev)

    def _merge_stats(self, other_count, other_mean, other_sum_sq_dev):
        if other_count == 0:
            return
        if self.count == 0:
            self.count = other_count
            self._mean = other_mean.copy()
            self._sum_sq_dev = other_sum_sq_dev.copy()
            return
        assert self._mean is not None and self._sum_sq_dev is not None
        total_count = self.count + other_count
        delta = other_mean - self._mean
        self._mean = self._mean + delta * (other_count / total_count)
        self._sum_sq_dev = (
            self._sum_sq_dev
            + other_sum_sq_dev
            + np.outer(delta, delta) * (self.count * other_count / total_count)
        )
        self.count = total_count

    def update(self, point):
        """Add a single data point.

        Args:
            point: A scalar, or a 1-D array-like of length `dim`.

        Returns:
            self, to allow chaining.
        """
        array = np.asarray(point, dtype=float)
        if array.ndim == 0:
            self._add_point(array.reshape(1))
        elif array.ndim == 1:
            self._add_point(array)
        else:
            raise ValueError(
                "update() expects a single point (scalar or 1-D "
                "array-like); use batch_update() for multiple points"
            )
        return self

    def batch_update(self, points):
        """Add a batch of data points.

        Args:
            points: A 1-D array-like of shape ``(n_points,)``, treated
                as a batch of scalar points (`dim` 1), or a 2-D
                array-like of shape ``(n_points, dim)``.

        Returns:
            self, to allow chaining.
        """
        array = np.asarray(points, dtype=float)
        if array.ndim == 1:
            self._merge_batch(array.reshape(-1, 1))
        elif array.ndim == 2:
            self._merge_batch(array)
        else:
            raise ValueError("batch_update() expects a 1-D or 2-D array-like")
        return self

    @property
    def mean(self):
        """Running mean vector, shape ``(dim,)``."""
        if self.count == 0:
            raise RuntimeError(
                "no data has been added yet; call update() or batch_update() first"
            )
        assert self._mean is not None
        return self._mean

    @property
    def cov(self):
        """Sample covariance matrix, shape ``(dim, dim)``."""
        if self.count == 0:
            raise RuntimeError(
                "no data has been added yet; call update() or batch_update() first"
            )
        if self.count - self.ddof <= 0:
            raise RuntimeError(
                f"need more than {self.ddof} point(s) to compute a covariance "
                f"with ddof={self.ddof}, but only {self.count} have been added"
            )
        assert self._sum_sq_dev is not None
        return self._sum_sq_dev / (self.count - self.ddof)

    @property
    def variance(self):
        """Per-dimension variance, i.e. the diagonal of `cov`."""
        return np.diag(self.cov)

    @property
    def std(self):
        """Per-dimension standard deviation."""
        return np.sqrt(self.variance)

    def copy(self):
        result = OnlineStats(dim=self.dim, ddof=self.ddof)
        result.count = self.count
        result._mean = None if self._mean is None else self._mean.copy()
        result._sum_sq_dev = (
            None if self._sum_sq_dev is None else self._sum_sq_dev.copy()
        )
        return result

    def __add__(self, other):
        if not isinstance(other, OnlineStats):
            return NotImplemented
        if self.dim is not None and other.dim is not None and self.dim != other.dim:
            raise ValueError(
                f"cannot combine OnlineStats of dimension {self.dim} "
                f"and {other.dim}"
            )
        result = self.copy()
        if other.count:
            result._merge_stats(other.count, other._mean, other._sum_sq_dev)
        elif result.dim is None:
            result.dim = other.dim
        return result

    def __iadd__(self, other):
        if isinstance(other, OnlineStats):
            if (
                self.dim is not None
                and other.dim is not None
                and self.dim != other.dim
            ):
                raise ValueError(
                    f"cannot combine OnlineStats of dimension {self.dim} "
                    f"and {other.dim}"
                )
            if other.count:
                self._merge_stats(other.count, other._mean, other._sum_sq_dev)
            elif self.dim is None:
                self.dim = other.dim
            return self
        try:
            array = np.asarray(other, dtype=float)
        except (TypeError, ValueError):
            return NotImplemented
        if array.ndim in (0, 1):
            self.update(array)
        elif array.ndim == 2:
            self.batch_update(array)
        else:
            return NotImplemented
        return self

    def __repr__(self):
        return f"OnlineStats(dim={self.dim}, count={self.count})"
