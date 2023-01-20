from collections import defaultdict
from typing import List

import numba
import numpy as np
import trackpy

from deeptangle.predict import Predictions


# The following lines injects a custom version of
# the Linker class into trackpy, which permits
# custom neighbor strategies.
class CustomLinker(trackpy.linking.linking.Linker):
    def __init__(self, search_range, memory=0, predictor=None,
                 adaptive_stop=None, adaptive_step=0.95,
                 neighbor_strategy=None, link_strategy=None,
                 dist_func=None, to_eucl=None, *args, **kwargs):
        neighbor_strategy_ = neighbor_strategy
        if not isinstance(neighbor_strategy, str):
            neighbor_strategy_ = 'BTree'
        super().__init__(search_range, memory=memory, predictor=predictor,
                 adaptive_stop=adaptive_stop, adaptive_step=adaptive_step,
                 neighbor_strategy=neighbor_strategy_, link_strategy=link_strategy,
                 dist_func=dist_func, to_eucl=to_eucl, *args, **kwargs)
        if not isinstance(neighbor_strategy, str):
            self.hash_cls = neighbor_strategy

trackpy.linking.linking.Linker = CustomLinker


class Query:
    """
    A class for querying a set of points using an asymmetrical distance function.

    The default query method in TrackPy assumes a symmetrical distance
    function, such as Euclidean distance. This class allows the use of
    an asymmetrical distance function.

    Parameters:
        points: A list of points.
        ndim: The number of dimensions of each point.
        dist_func: A distance function that takes two points as arguments and returns a
                   distance. The distance function must be decorated with `@numba.jit`
                   if it is not already compiled by Numba.

    Attributes:
        points: A list of points.
        ndim: The number of dimensions of each point.
        dist_func: A distance function that takes two points as arguments and returns a distance.
        coords_mapped: A list of the positions of the points in `self.points`.
    """

    def __init__(self, points, ndim, dist_func, **kwargs):
        self.points = points
        self.ndim = ndim
        self.dist_func = dist_func
        self.coords_mapped = [x.pos for x in points]

    def query(self, coords, max_neighbors, rescale, search_range):
        """
        Query the points for the nearest neighbors of the given coordinates.

        Parameters:
            coords: The coordinates to query.
            max_neighbors: The maximum number of nearest neighbors to return for each coordinate.
            rescale: Whether to rescale the distances.
            search_range: The maximum distance to search for nearest neighbors. Distances
                          greater than this value will be set to infinity.

        Returns:
            dists: An array of shape (len(coords), max_neighbors) containing the
                   sorted distances to the nearest neighbors.
            indices: An array of shape (len(coords), max_neighbors) containing the
                     indices of the nearest neighbors in `self.points`.
        """
        dists = np.zeros((len(coords), len(self.coords_mapped)), dtype=np.float64)

        for i in range(len(coords)):
            for j in range(len(self.coords_mapped)):
                d = self.dist_func(coords[i], self.coords_mapped[j])
                dists[i, j] = d

        indices = np.argsort(dists, axis=1)
        dists.sort(axis=1)

        indices, dists = indices[:, :max_neighbors], dists[:, :max_neighbors]
        dists[dists > search_range] = np.inf
        return dists, indices

    def add_point(self, pt):
        self.points.append(pt)
        self.coords_mapped.append(pt.pos)


@numba.njit
def directed_distance(future, current):
    ls_dim = 8
    spatial_cutoff_sq = 25**2

    # Midpoint filtering.
    midpoint_ind = (len(current) - ls_dim) // 2
    current_midpoint = current[midpoint_ind - 1 : midpoint_ind + 1]
    future_midpoint = future[midpoint_ind - 1 : midpoint_ind + 1]
    if ((current_midpoint - future_midpoint) ** 2).sum(-1) >= spatial_cutoff_sq:
        return np.inf

    # We compare them by reshaping them to be 3D arrays. where past-present,
    # and present-future are aligned.
    xa = current[:-ls_dim].reshape(3, -1, 2)[1:]
    xb = future[:-ls_dim].reshape(3, -1, 2)[:-1]

    # We define a function that computes the distance between splines.
    sdist = lambda x, y: np.sum(np.sum((x - y) ** 2, axis=-1), axis=-1)

    # The minimal distance between the flipped version and the straight is used.
    xa_flip = xa[:, ::-1]
    distance = np.minimum(sdist(xa, xb), sdist(xa_flip, xb))

    # The sum of the distances is returned.
    return np.sum(np.sqrt(distance))


def identity_assignment(
    predictions: List[Predictions],
    search_range: float = 500,
    adaptive_step: float = 0.9,
    memory: int = 15,
):
    """
    Performs identity assignment between consecutive predictions.

    Parameters:
        predictions: List of consecutive predicitons.
        search_range: The maximum distance to search for nearest neighbors. Distances
                      greater than this value will be set to infinity.
        adaptive_step: Reduce search_range by multiplying it by this factor.
        memory: The maximum number of frames during which a feature can vanish, then reappear nearby,
                and be considered the same particle. 0 by default.

    Returns:
        particles: A list with the idenitites assigned to each spline.
        splines: A list with the splines detected in each frame.
    """

    def transform(pred):
        N = len(pred.w)
        return np.concatenate((pred.w.reshape(N, -1), pred.p), axis=-1)

    flatten_predictions = list(map(transform, predictions))

    links = trackpy.link_iter(
        flatten_predictions,
        search_range=search_range,
        adaptive_stop=1,
        adaptive_step=adaptive_step,
        dist_func=directed_distance,
        memory=memory,
        neighbor_strategy=Query,
    )
    particles = [p for (_, p) in links]
    splines = [p.w[:, 1] for p in predictions]
    return particles, splines


def _find_endpoints(tracks, X, window=5, low=32, high=480):
    results = {"start": [], "end": []}

    is_near_border = lambda x: (x < low).any() or (x > high).any()

    for i in range(window, len(tracks) - window):
        current_tracks = tracks[i]
        other_tracks = {"past": [], "future": []}
        for j in range(1, window + 1):
            other_tracks["past"].extend(tracks[i - j])
            other_tracks["future"].extend(tracks[i + j])

        for j, track in enumerate(current_tracks):
            if track not in other_tracks["past"]:
                if not is_near_border(X[i][j]):
                    results["start"].append((i, track, *X[i][j][25]))
            elif track not in other_tracks["future"]:
                if not is_near_border(X[i][j]):
                    results["end"].append((i, track, *X[i][j][25]))
    return results


def _register_pair(d, p1, p2):
    p1, p2 = sorted((p1, p2))
    d[p2] = p1 if p1 not in d else d[p1]
    return d


def _replace_tracks(tracks, pairs):
    new_tracks = []
    for current_tracks in tracks:
        new_tracks.append([pairs[t] if t in pairs else t for t in current_tracks])
    return new_tracks


def _pair_mapping(endpoints, cutoff=4, threshold=20):
    end, start = np.array(endpoints["end"]), np.array(endpoints["start"])

    if len(end) == 0 or len(start) == 0:
        return {}

    # Only those that are closeby in time and position can be candidates.
    time_candidates = np.abs(end[:, None, 0] - start[None, :, 0]) < cutoff
    pos_candidates = np.sqrt(np.sum((end[:, None, 2:] - start[None, :, 2:]) ** 2, -1)) <= threshold

    end_idx, start_idx = np.nonzero(time_candidates * pos_candidates)
    pair_dict = {}
    for i, j in zip(end_idx, start_idx):
        pair_dict = _register_pair(pair_dict, int(end[i, 1]), int(start[j, 1]))
    return pair_dict


def _filter_stubs(tracks, values, minimum=5):
    counter = defaultdict(int)
    for p in tracks:
        for pi in p:
            counter[pi] += 1

    new_tracks = []
    new_values = []
    for i, p in enumerate(tracks):
        _new_tracks = []
        _new_values = []
        for j, pi in enumerate(p):
            if counter[pi] < minimum:
                continue
            if pi in _new_tracks:
                continue
            _new_tracks.append(pi)
            _new_values.append(values[i][j])
        new_tracks.append(_new_tracks)
        new_values.append(np.array(_new_values))
    return new_tracks, new_values


def merge_tracks(tracks, values, framesize=512, padding=32):
    # TODO(albert): Assumes square frames, and it shouldn't
    endpoints = _find_endpoints(tracks, values, low=padding, high=framesize - padding)
    pairs_map = _pair_mapping(endpoints, cutoff=8)
    new_tracks = _replace_tracks(tracks, pairs_map)
    new_tracks, new_values = _filter_stubs(new_tracks, values)
    return new_tracks, new_values
