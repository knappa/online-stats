"""Online mean and covariance statistics via the Welford/Chan algorithm.

References:
    Knuth, D. E. *The Art of Computer Programming, Volume 2:
    Seminumerical Algorithms*, section 4.2.2.
    Chan, T. F. et al. "Updating Formulae and a Pairwise Algorithm for
    Computing Sample Variances." (1979).
"""

from ._online_stats import OnlineStats

__all__ = ["OnlineStats"]
