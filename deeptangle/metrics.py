import numba
import numpy as np

@numba.njit
def asymmetric_dtw(a, b):
    """
    Computes the asymmetric dynamic time warping metric between two discrete curves.
    len(a) > len(b) and the metric is computed for the values of b.
    """
    n, m = len(a), len(b)
    cost_matrix = np.full(shape=(n-1, m), fill_value=1e8)

    dist2 = lambda x, y: np.sum((x-y)**2)
    def dist_to_line(endpoints, p):
        """Distance between a point in the segment [v,w] and a point p."""
        v, w = endpoints
        t = min(max(np.dot(p-v, w-v) / dist2(v, w), 0.0), 1.0)
        line_p = v + t*(w-v)
        return np.sqrt(dist2(p, line_p))

    segments = np.stack((a[:-1], a[1:]), axis=1)

    d = np.zeros(shape=(n-1, m))
    for i in range(n-1):
        for j in range(m):
            d[i, j] = dist_to_line(segments[i], b[j])

    cost_matrix[0,0] = d[0,0]
    for j in range(1, m):
        cost_matrix[0,j] = cost_matrix[0,j-1] + d[0,j]

    for i in range(1, n-1):
        cost_matrix[i,0] = min(cost_matrix[i-1,0], d[i,0])

    for i in range(1, n-1):
        for j in range(1, m):
            cost_matrix[i,j] = min(cost_matrix[i-1,j], cost_matrix[i,j-1]+d[i,j])

    return cost_matrix[-1, -1] / m
