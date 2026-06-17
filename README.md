# online-stats

Online mean and covariance statistics using Welford's algorithm
(Knuth, *The Art of Computer Programming*, Vol. 2), generalized to
vector-valued data and to combining independent batches (Chan et al.).

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
