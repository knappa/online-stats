# online-stats

Online mean and covariance statistics using Welford's algorithm
[[Welford 1962]](#references), generalized to vector-valued data and
to combining independent batches [[Chan et al. 1979]](#references).
See also Knuth's presentation of the running-mean recurrence
[[Knuth 1997]](#references).

```python
from online_stats import OnlineStats

stats = OnlineStats([[3.0, 4.0], [5.0, 6.0]])  # initialize from a batch
stats.update([1.0, 2.0])                       # one point
stats.batch_update([[7.0, 8.0], [9.0, 10.0]])  # another batch

stats.mean   # running mean vector
stats.cov    # running sample covariance matrix
stats.count  # number of points seen

a = OnlineStats()
a.update([1.0, 2.0])
b = OnlineStats()
b.update([3.0, 4.0])
combined = a + b  # merge two batches' statistics
```

## References

- Welford, B. P. (1962). Note on a Method for Calculating Corrected
  Sums of Squares and Products. *Technometrics*, 4(3), 419-420.
  https://doi.org/10.1080/00401706.1962.10490022
- Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). Algorithms for
  Computing the Sample Variance: Analysis and Recommendations.
  *The American Statistician*, 37(3), 242-247.
  https://doi.org/10.1080/00031305.1983.10483115
- Knuth, D. E. (1997). *The Art of Computer Programming, Volume 2:
  Seminumerical Algorithms* (3rd ed.). Addison-Wesley.
