"""Exact 2D Sibson natural-neighbor interpolation (Numba-accelerated).

Computes analytical Voronoi area-stealing weights using circumcenter geometry,
matching MATLAB's scatteredInterpolant(..., 'natural') behavior.

Core loops are JIT-compiled with Numba for performance comparable to MATLAB.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange, types
from numba.typed import List as NumbaList
from scipy.spatial import Delaunay


# ── Numba-compiled core functions ──

@njit(cache=True)
def _circumcenter_nb(ax, ay, bx, by, cx, cy):
    """Circumcenter of triangle (a,b,c). Returns (ux, uy, valid)."""
    D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-30:
        return 0.0, 0.0, False
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / D
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / D
    return ux, uy, True


@njit(cache=True)
def _polygon_area_nb(vx, vy, n):
    """Absolute area of polygon with n vertices using shoelace formula."""
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vx[i] * vy[j] - vy[i] * vx[j]
    return abs(area) * 0.5


@njit(cache=True)
def _find_cavity_nb(simplices, neighbors, points_x, points_y, qx, qy, start_simplex):
    """Find all simplices whose circumcircle contains query point (qx, qy).
    Returns boolean mask over simplices."""
    n_simp = len(simplices)
    in_cavity = np.zeros(n_simp, dtype=np.bool_)

    stack = np.empty(n_simp, dtype=np.int64)
    stack_top = 0
    stack[0] = start_simplex
    stack_top = 1

    while stack_top > 0:
        stack_top -= 1
        s = stack[stack_top]
        if s < 0 or s >= n_simp or in_cavity[s]:
            continue
        i0, i1, i2 = simplices[s, 0], simplices[s, 1], simplices[s, 2]
        ccx, ccy, valid = _circumcenter_nb(
            points_x[i0], points_y[i0],
            points_x[i1], points_y[i1],
            points_x[i2], points_y[i2])
        if not valid:
            continue
        r2 = (points_x[i0] - ccx) ** 2 + (points_y[i0] - ccy) ** 2
        d2 = (qx - ccx) ** 2 + (qy - ccy) ** 2
        if d2 <= r2 * 1.0000000001:
            in_cavity[s] = True
            for k in range(3):
                nb = neighbors[s, k]
                if nb >= 0 and not in_cavity[nb]:
                    stack[stack_top] = nb
                    stack_top += 1

    return in_cavity


@njit(cache=True)
def _sibson_one_query(simplices, neighbors, points_x, points_y,
                      all_ccx, all_ccy, all_cc_valid,
                      qx, qy, start_simplex,
                      values, n_pts):
    """Compute Sibson-interpolated value for one query point.

    Returns (value, valid_flag).
    """
    n_simp = len(simplices)
    in_cavity = _find_cavity_nb(simplices, neighbors, points_x, points_y,
                                qx, qy, start_simplex)

    # count cavity triangles
    cavity_count = 0
    for s in range(n_simp):
        if in_cavity[s]:
            cavity_count += 1
    if cavity_count == 0:
        return 0.0, False

    # find boundary edges: edges of cavity triangles that border non-cavity
    # store as (va, vb, simplex_index)
    max_edges = cavity_count * 3
    edge_va = np.empty(max_edges, dtype=np.int64)
    edge_vb = np.empty(max_edges, dtype=np.int64)
    edge_src = np.empty(max_edges, dtype=np.int64)
    n_edges = 0

    for s in range(n_simp):
        if not in_cavity[s]:
            continue
        for i in range(3):
            nb = neighbors[s, i]
            if nb < 0 or not in_cavity[nb]:
                j = (i + 1) % 3
                k = (i + 2) % 3
                edge_va[n_edges] = simplices[s, j]
                edge_vb[n_edges] = simplices[s, k]
                edge_src[n_edges] = s
                n_edges += 1

    if n_edges < 3:
        return 0.0, False

    # order boundary edges into a cycle
    ordered_va = np.empty(n_edges, dtype=np.int64)
    ordered_vb = np.empty(n_edges, dtype=np.int64)
    ordered_src = np.empty(n_edges, dtype=np.int64)
    used = np.zeros(n_edges, dtype=np.bool_)

    ordered_va[0] = edge_va[0]
    ordered_vb[0] = edge_vb[0]
    ordered_src[0] = edge_src[0]
    used[0] = True
    n_ordered = 1

    for _ in range(n_edges - 1):
        last_vb = ordered_vb[n_ordered - 1]
        found = False
        for idx in range(n_edges):
            if used[idx]:
                continue
            if edge_va[idx] == last_vb:
                ordered_va[n_ordered] = edge_va[idx]
                ordered_vb[n_ordered] = edge_vb[idx]
                ordered_src[n_ordered] = edge_src[idx]
                used[idx] = True
                n_ordered += 1
                found = True
                break
            elif edge_vb[idx] == last_vb:
                ordered_va[n_ordered] = edge_vb[idx]
                ordered_vb[n_ordered] = edge_va[idx]
                ordered_src[n_ordered] = edge_src[idx]
                used[idx] = True
                n_ordered += 1
                found = True
                break
        if not found:
            break

    if n_ordered != n_edges:
        return 0.0, False

    # compute new circumcenters: cc(q, va[i], vb[i])
    new_ccx = np.empty(n_ordered, dtype=np.float64)
    new_ccy = np.empty(n_ordered, dtype=np.float64)
    for i in range(n_ordered):
        va = ordered_va[i]
        vb = ordered_vb[i]
        cx, cy, valid = _circumcenter_nb(qx, qy,
                                         points_x[va], points_y[va],
                                         points_x[vb], points_y[vb])
        if not valid:
            return 0.0, False
        new_ccx[i] = cx
        new_ccy[i] = cy

    # for each boundary vertex (= natural neighbor), compute stolen area
    # boundary_verts = ordered_va[0], ordered_va[1], ..., ordered_va[n-1]
    total_weight = 0.0
    weighted_val = 0.0
    poly_x = np.empty(cavity_count + 2, dtype=np.float64)
    poly_y = np.empty(cavity_count + 2, dtype=np.float64)

    for i in range(n_ordered):
        vi = ordered_va[i]
        prev_i = (i - 1) % n_ordered

        # walk cavity triangles containing vi from src[prev_i] to src[i]
        tri_start = ordered_src[prev_i]
        tri_end = ordered_src[i]

        # build stolen polygon: new_cc[prev_i], old_ccs..., new_cc[i]
        n_poly = 0
        poly_x[n_poly] = new_ccx[prev_i]
        poly_y[n_poly] = new_ccy[prev_i]
        n_poly += 1

        # walk fan of cavity triangles around vi
        current = tri_start
        visited = np.zeros(n_simp, dtype=np.bool_)
        visited[current] = True

        if all_cc_valid[current]:
            poly_x[n_poly] = all_ccx[current]
            poly_y[n_poly] = all_ccy[current]
            n_poly += 1

        max_walk = cavity_count + 1
        for _ in range(max_walk):
            if current == tri_end and n_poly > 1:
                break
            found_next = False
            for nb_k in range(3):
                nb = neighbors[current, nb_k]
                if nb < 0 or visited[nb] or not in_cavity[nb]:
                    continue
                # check if nb contains vi
                has_vi = False
                for vv in range(3):
                    if simplices[nb, vv] == vi:
                        has_vi = True
                        break
                if has_vi:
                    visited[nb] = True
                    current = nb
                    if all_cc_valid[current]:
                        poly_x[n_poly] = all_ccx[current]
                        poly_y[n_poly] = all_ccy[current]
                        n_poly += 1
                    found_next = True
                    break
            if not found_next:
                break

        poly_x[n_poly] = new_ccx[i]
        poly_y[n_poly] = new_ccy[i]
        n_poly += 1

        if n_poly >= 3:
            area = _polygon_area_nb(poly_x, poly_y, n_poly)
            total_weight += area
            weighted_val += area * values[vi]

    if total_weight < 1e-30:
        return 0.0, False

    return weighted_val / total_weight, True


@njit(cache=True)
def _sibson_batch(simplices, neighbors, points_x, points_y,
                  all_ccx, all_ccy, all_cc_valid,
                  query_x, query_y, start_simplices,
                  values, n_pts):
    """Batch Sibson interpolation for all query points."""
    M = len(query_x)
    result = np.full(M, np.nan)

    for qi in range(M):
        s0 = start_simplices[qi]
        if s0 < 0:
            continue
        val, ok = _sibson_one_query(
            simplices, neighbors, points_x, points_y,
            all_ccx, all_ccy, all_cc_valid,
            query_x[qi], query_y[qi], s0,
            values, n_pts)
        if ok:
            result[qi] = val

    return result


# ── Public API ──

def sibson_interp(points, values, query_points):
    """Exact Sibson natural-neighbor interpolation in 2D (Numba-accelerated).

    Parameters
    ----------
    points : (N, 2) — data point coordinates
    values : (N,) — data values
    query_points : (M, 2) — query coordinates

    Returns
    -------
    result : (M,) — interpolated values (NaN outside convex hull)
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)

    # handle duplicate points
    from collections import defaultdict
    pt_map = defaultdict(list)
    for i in range(len(points)):
        key = (round(points[i, 0], 6), round(points[i, 1], 6))
        pt_map[key].append(i)

    if len(pt_map) < len(points):
        unique_pts = []
        unique_vals = []
        for key, indices in pt_map.items():
            unique_pts.append(points[indices[0]])
            unique_vals.append(np.mean(values[indices]))
        points = np.array(unique_pts)
        values = np.array(unique_vals)

    tri = Delaunay(points)
    px = np.ascontiguousarray(points[:, 0])
    py = np.ascontiguousarray(points[:, 1])
    simpl = np.ascontiguousarray(tri.simplices.astype(np.int64))
    nbrs = np.ascontiguousarray(tri.neighbors.astype(np.int64))

    # pre-compute circumcenters
    n_simp = len(simpl)
    all_ccx = np.zeros(n_simp)
    all_ccy = np.zeros(n_simp)
    all_cc_valid = np.zeros(n_simp, dtype=np.bool_)
    for s in range(n_simp):
        i0, i1, i2 = simpl[s]
        cx, cy, ok = _circumcenter_nb(px[i0], py[i0], px[i1], py[i1], px[i2], py[i2])
        if ok:
            all_ccx[s] = cx
            all_ccy[s] = cy
            all_cc_valid[s] = True

    # batch find_simplex (vectorized in C)
    start_simplices = tri.find_simplex(query_points).astype(np.int64)

    qx = np.ascontiguousarray(query_points[:, 0])
    qy = np.ascontiguousarray(query_points[:, 1])

    return _sibson_batch(simpl, nbrs, px, py,
                         all_ccx, all_ccy, all_cc_valid,
                         qx, qy, start_simplices,
                         values, len(points))


@njit(cache=True)
def _sibson_weights_one(simplices, neighbors, points_x, points_y,
                        all_ccx, all_ccy, all_cc_valid,
                        qx, qy, start_simplex, n_pts):
    """Compute Sibson weights for one query point.

    Returns (neighbor_indices, weights, n_neighbors).
    Max neighbors is bounded by ~20 for typical Delaunay.
    """
    MAX_NB = 64
    nb_idx = np.empty(MAX_NB, dtype=np.int64)
    nb_wgt = np.empty(MAX_NB, dtype=np.float64)

    n_simp = len(simplices)
    in_cavity = _find_cavity_nb(simplices, neighbors, points_x, points_y,
                                qx, qy, start_simplex)

    cavity_count = 0
    for s in range(n_simp):
        if in_cavity[s]:
            cavity_count += 1
    if cavity_count == 0:
        return nb_idx, nb_wgt, 0

    max_edges = cavity_count * 3
    edge_va = np.empty(max_edges, dtype=np.int64)
    edge_vb = np.empty(max_edges, dtype=np.int64)
    edge_src = np.empty(max_edges, dtype=np.int64)
    n_edges = 0
    for s in range(n_simp):
        if not in_cavity[s]:
            continue
        for i in range(3):
            nb = neighbors[s, i]
            if nb < 0 or not in_cavity[nb]:
                j = (i + 1) % 3
                k = (i + 2) % 3
                edge_va[n_edges] = simplices[s, j]
                edge_vb[n_edges] = simplices[s, k]
                edge_src[n_edges] = s
                n_edges += 1
    if n_edges < 3:
        return nb_idx, nb_wgt, 0

    ordered_va = np.empty(n_edges, dtype=np.int64)
    ordered_vb = np.empty(n_edges, dtype=np.int64)
    ordered_src = np.empty(n_edges, dtype=np.int64)
    used = np.zeros(n_edges, dtype=np.bool_)
    ordered_va[0] = edge_va[0]
    ordered_vb[0] = edge_vb[0]
    ordered_src[0] = edge_src[0]
    used[0] = True
    n_ordered = 1
    for _ in range(n_edges - 1):
        last_vb = ordered_vb[n_ordered - 1]
        found = False
        for idx in range(n_edges):
            if used[idx]:
                continue
            if edge_va[idx] == last_vb:
                ordered_va[n_ordered] = edge_va[idx]
                ordered_vb[n_ordered] = edge_vb[idx]
                ordered_src[n_ordered] = edge_src[idx]
                used[idx] = True
                n_ordered += 1
                found = True
                break
            elif edge_vb[idx] == last_vb:
                ordered_va[n_ordered] = edge_vb[idx]
                ordered_vb[n_ordered] = edge_va[idx]
                ordered_src[n_ordered] = edge_src[idx]
                used[idx] = True
                n_ordered += 1
                found = True
                break
        if not found:
            break
    if n_ordered != n_edges:
        return nb_idx, nb_wgt, 0

    new_ccx = np.empty(n_ordered, dtype=np.float64)
    new_ccy = np.empty(n_ordered, dtype=np.float64)
    for i in range(n_ordered):
        va = ordered_va[i]
        vb = ordered_vb[i]
        cx, cy, valid = _circumcenter_nb(qx, qy,
                                         points_x[va], points_y[va],
                                         points_x[vb], points_y[vb])
        if not valid:
            return nb_idx, nb_wgt, 0
        new_ccx[i] = cx
        new_ccy[i] = cy

    poly_x = np.empty(cavity_count + 2, dtype=np.float64)
    poly_y = np.empty(cavity_count + 2, dtype=np.float64)
    total_weight = 0.0
    n_nb = 0

    for i in range(n_ordered):
        vi = ordered_va[i]
        prev_i = (i - 1) % n_ordered
        tri_start = ordered_src[prev_i]
        tri_end = ordered_src[i]

        n_poly = 0
        poly_x[n_poly] = new_ccx[prev_i]
        poly_y[n_poly] = new_ccy[prev_i]
        n_poly += 1

        current = tri_start
        visited = np.zeros(n_simp, dtype=np.bool_)
        visited[current] = True
        if all_cc_valid[current]:
            poly_x[n_poly] = all_ccx[current]
            poly_y[n_poly] = all_ccy[current]
            n_poly += 1

        for _ in range(cavity_count + 1):
            if current == tri_end and n_poly > 1:
                break
            found_next = False
            for nb_k in range(3):
                nb = neighbors[current, nb_k]
                if nb < 0 or visited[nb] or not in_cavity[nb]:
                    continue
                has_vi = False
                for vv in range(3):
                    if simplices[nb, vv] == vi:
                        has_vi = True
                        break
                if has_vi:
                    visited[nb] = True
                    current = nb
                    if all_cc_valid[current]:
                        poly_x[n_poly] = all_ccx[current]
                        poly_y[n_poly] = all_ccy[current]
                        n_poly += 1
                    found_next = True
                    break
            if not found_next:
                break

        poly_x[n_poly] = new_ccx[i]
        poly_y[n_poly] = new_ccy[i]
        n_poly += 1

        if n_poly >= 3:
            area = _polygon_area_nb(poly_x, poly_y, n_poly)
            if n_nb < MAX_NB:
                nb_idx[n_nb] = vi
                nb_wgt[n_nb] = area
                total_weight += area
                n_nb += 1

    if total_weight > 1e-30:
        for k in range(n_nb):
            nb_wgt[k] /= total_weight

    return nb_idx, nb_wgt, n_nb


MAX_NB_PER_QUERY = 20


@njit(parallel=True, cache=True)
def _build_weight_matrix_data(simplices, neighbors, points_x, points_y,
                              all_ccx, all_ccy, all_cc_valid,
                              query_x, query_y, start_simplices, n_pts):
    """Build sparse weight matrix data (COO format), parallelized over queries."""
    M = len(query_x)
    # Pre-allocate per-query arrays (fixed-size, max ~20 neighbors each)
    all_nb_idx = np.empty((M, MAX_NB_PER_QUERY), dtype=np.int64)
    all_nb_wgt = np.empty((M, MAX_NB_PER_QUERY), dtype=np.float64)
    all_n_nb = np.zeros(M, dtype=np.int64)

    for qi in prange(M):
        s0 = start_simplices[qi]
        if s0 < 0:
            continue
        nb_idx, nb_wgt, n_nb = _sibson_weights_one(
            simplices, neighbors, points_x, points_y,
            all_ccx, all_ccy, all_cc_valid,
            query_x[qi], query_y[qi], s0, n_pts)
        n = min(n_nb, MAX_NB_PER_QUERY)
        all_n_nb[qi] = n
        for k in range(n):
            all_nb_idx[qi, k] = nb_idx[k]
            all_nb_wgt[qi, k] = nb_wgt[k]

    # Count total nnz and build COO arrays
    total_nnz = 0
    for qi in range(M):
        total_nnz += all_n_nb[qi]

    rows = np.empty(total_nnz, dtype=np.int64)
    cols = np.empty(total_nnz, dtype=np.int64)
    data = np.empty(total_nnz, dtype=np.float64)
    pos = 0
    for qi in range(M):
        n = all_n_nb[qi]
        for k in range(n):
            rows[pos] = qi
            cols[pos] = all_nb_idx[qi, k]
            data[pos] = all_nb_wgt[qi, k]
            pos += 1

    return rows, cols, data


def sibson_weight_matrix(points, query_points):
    """Precompute Sibson weight matrix as scipy sparse CSR.

    Parameters
    ----------
    points : (N, 2) — deduplicated station coordinates
    query_points : (M, 2) — grid query coordinates

    Returns
    -------
    W : scipy.sparse.csr_matrix (M, N) — weight matrix
        result = W @ values gives interpolated values
    """
    from scipy.sparse import coo_matrix

    points = np.asarray(points, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)

    tri = Delaunay(points)
    px = np.ascontiguousarray(points[:, 0])
    py = np.ascontiguousarray(points[:, 1])
    simpl = np.ascontiguousarray(tri.simplices.astype(np.int64))
    nbrs = np.ascontiguousarray(tri.neighbors.astype(np.int64))

    n_simp = len(simpl)
    all_ccx = np.zeros(n_simp)
    all_ccy = np.zeros(n_simp)
    all_cc_valid = np.zeros(n_simp, dtype=np.bool_)
    for s in range(n_simp):
        i0, i1, i2 = simpl[s]
        cx, cy, ok = _circumcenter_nb(px[i0], py[i0], px[i1], py[i1], px[i2], py[i2])
        if ok:
            all_ccx[s] = cx
            all_ccy[s] = cy
            all_cc_valid[s] = True

    start_simplices = tri.find_simplex(query_points).astype(np.int64)
    qx = np.ascontiguousarray(query_points[:, 0])
    qy = np.ascontiguousarray(query_points[:, 1])

    rows, cols, data = _build_weight_matrix_data(
        simpl, nbrs, px, py, all_ccx, all_ccy, all_cc_valid,
        qx, qy, start_simplices, len(points))

    M = len(query_points)
    N = len(points)
    W = coo_matrix((data, (rows, cols)), shape=(M, N)).tocsr()
    return W
