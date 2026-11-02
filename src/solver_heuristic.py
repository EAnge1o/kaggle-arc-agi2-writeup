"""Heuristic rule-battery solver for ARC-AGI-2.

Philosophy (per user guidance): do NOT chase the optimal program. Apply a
battery of empirical transformation rules in rough order of expected yield,
verify each rule against ALL training pairs exactly, and apply the first
consistent rule to the test input. Fall back to output-size prediction +
content-preserving guesses when no rule fits.

Rules are pure functions (input_grid, params) -> grid; params are inferred
from train pairs. Each rule returns a list of candidate params to try.
"""
from __future__ import annotations

import numpy as np
from collections import Counter
from itertools import permutations

GRID = list[list[int]]


def as_np(g):
    return np.array(g, dtype=np.int64)


def as_list(a):
    return a.tolist()


# --------------------------------------------------------------------------
# basic geometric ops (param-free)
# --------------------------------------------------------------------------

def op_identity(a): return a
def op_hflip(a): return a[:, ::-1]
def op_vflip(a): return a[::-1, :]
def op_transpose(a): return a.T
def op_rot90(a): return np.rot90(a, 1)
def op_rot180(a): return np.rot90(a, 2)
def op_rot270(a): return np.rot90(a, 3)
def op_dmirror(a): return a[::-1, ::-1].T

GEO_OPS = [("identity", op_identity), ("hflip", op_hflip), ("vflip", op_vflip),
           ("transpose", op_transpose), ("rot90", op_rot90),
           ("rot180", op_rot180), ("rot270", op_rot270), ("dmirror", op_dmirror)]


# --------------------------------------------------------------------------
# color mapping helpers
# --------------------------------------------------------------------------

def learn_color_map(pairs):
    """Learn a consistent global color mapping from (input,output) pairs.

    First pass: same-position cell correspondences. Second pass: for colors
    never observed, match by frequency. Returns dict or None.
    """
    seen: dict[int, int] = {}
    freq_in, freq_out = Counter(), Counter()
    for inp, out in pairs:
        a, b = as_np(inp), as_np(out)
        if a.shape != b.shape:
            return None
        for x, y in zip(a.ravel(), b.ravel()):
            freq_in[int(x)] += 1
            freq_out[int(y)] += 1
            if int(x) in seen and seen[int(x)] != int(y):
                return None
            seen[int(x)] = int(y)
    # check injectivity of seen
    vals = list(seen.values())
    if len(set(vals)) != len(vals):
        return None
    # fill unobserved colors by frequency matching
    unmapped_in = [c for c in sorted(freq_in) if c not in seen]
    unmapped_out = [c for c in sorted(freq_out) if c not in vals]
    if len(unmapped_in) == len(unmapped_out):
        # order by frequency (desc) then color id
        order_in = sorted(unmapped_in, key=lambda c: (-freq_in[c], c))
        order_out = sorted(unmapped_out, key=lambda c: (-freq_out[c], c))
        for x, y in zip(order_in, order_out):
            seen[x] = y
    return seen


# --------------------------------------------------------------------------
# rule candidates: each returns a list of (param_tag, transform_fn) options
# --------------------------------------------------------------------------

def candidates_geometry(pairs):
    out = []
    for name, fn in GEO_OPS:
        if all(fn(as_np(inp)).tolist() == out for inp, out in pairs):
            out.append((name, lambda a, f=fn: f(a)))
    return out


def candidates_color_map(pairs):
    cm = learn_color_map(pairs)
    out = []
    if cm is None:
        return out
    # plain remap
    if all(
        as_list(np.vectorize(lambda x: cm.get(int(x), int(x)))(as_np(inp))) == out
        for inp, out in pairs
    ):
        out.append(("colormap", lambda a, cm=cm: as_list(
            np.vectorize(lambda x: cm.get(int(x), int(x)))(a))))
    # geometry followed by remap
    for name, fn in GEO_OPS:
        if all(
            as_list(np.vectorize(lambda x: cm.get(int(x), int(x)))(fn(as_np(inp)))) == out
            for inp, out in pairs
        ):
            out.append((f"{name}+colormap",
                        lambda a, f=fn, cm=cm: as_list(
                            np.vectorize(lambda x: cm.get(int(x), int(x)))(f(a)))))
    return out


def candidates_upscale(pairs):
    out = []
    for k in range(2, 7):
        ok = True
        for inp, out in pairs:
            a = as_np(inp)
            if (a.shape[0] * k != len(out)) or (a.shape[1] * k != len(out[0])):
                ok = False
                break
        if not ok:
            continue
        # cell -> kxk block
        if all(
            as_list(np.repeat(np.repeat(as_np(inp), k, axis=0), k, axis=1)) == out
            for inp, out in pairs
        ):
            out.append(("upscale%d" % k,
                        lambda a, k=k: as_list(np.repeat(np.repeat(a, k, axis=0), k, axis=1))))
    return out


def candidates_downscale(pairs):
    out = []
    for k in range(2, 7):
        ok = True
        for inp, out in pairs:
            a = as_np(inp)
            if (len(out) * k != a.shape[0]) or (len(out[0]) * k != a.shape[1]):
                ok = False
                break
        if not ok:
            continue
        good = True
        for inp, out in pairs:
            a = as_np(inp)
            h, w = len(out), len(out[0])
            blocks = a.reshape(h, k, w, k)
            for i in range(h):
                for j in range(w):
                    if len(np.unique(blocks[i, :, j, :])) != 1:
                        good = False
                        break
                if not good:
                    break
            if not good:
                break
        if good and all(
            as_list(np.array([[as_np(inp).reshape(len(out), k, len(out[0]), k)[i, 0, j, 0]
                                for j in range(len(out[0]))]
                               for i in range(len(out))])) == out
            for inp, out in pairs
        ):
            out.append(("downscale%d" % k,
                        lambda a, k=k: as_list(
                            a.reshape(a.shape[0] // k, k, a.shape[1] // k, k)[:, 0, :, 0])))
    return out


def candidates_crop(pairs):
    out = []
    for inp0, out0 in pairs[:1]:
        a, b = as_np(inp0), as_np(out0)
        h, w = b.shape
        if h > a.shape[0] or w > a.shape[1]:
            return out
        for top in range(a.shape[0] - h + 1):
            for left in range(a.shape[1] - w + 1):
                if not (a[top:top + h, left:left + w] == b).all():
                    continue
                if all(as_np(inp)[top:top + h, left:left + w].tolist() == out
                       for inp, out in pairs):
                    out.append((f"crop({top},{left})",
                                lambda a, t=top, l=left, h=h, w=w: a[t:t + h, l:l + w]))
    return out


def candidates_bbox(pairs):
    """output = input clipped to bbox of non-background cells (bg = most common color)."""
    for inp, out in pairs:
        a = as_np(inp)
        bg = Counter(a.ravel().tolist()).most_common(1)[0][0]
        nz = np.argwhere(a != bg)
        if len(nz) == 0:
            return []
        y0, x0 = nz.min(0)
        y1, x1 = nz.max(0)
        if not (a[y0:y1 + 1, x0:x1 + 1] == as_np(out)).all():
            return []
    def f(a):
        bg = Counter(a.ravel().tolist()).most_common(1)[0][0]
        nz = np.argwhere(a != bg)
        if len(nz) == 0:
            return a.tolist()
        y0, x0 = nz.min(0)
        y1, x1 = nz.max(0)
        return a[y0:y1 + 1, x0:x1 + 1].tolist()
    return [("bbox", f)]


def candidates_pad(pairs):
    """input pasted into a larger canvas (offset consistent across pairs)."""
    out = []
    for inp0, out0 in pairs[:1]:
        a, b = as_np(inp0), as_np(out0)
        h, w = a.shape
        if h > b.shape[0] or w > b.shape[1]:
            return out
        for top in range(b.shape[0] - h + 1):
            for left in range(b.shape[1] - w + 1):
                canvas = np.zeros_like(b)
                canvas[top:top + h, left:left + w] = a
                if not (canvas == b).all():
                    continue
                if all(
                    (lambda c: (c[:] == b) or True)(
                        np.pad(as_np(inp),
                               ((top, len(out) - top - h), (left, len(out[0]) - left - w))))
                    for inp, out in pairs
                ):
                    pass
                # re-verify properly
                ok = True
                for inp, out in pairs:
                    c = np.zeros(as_np(out).shape, dtype=np.int64)
                    c[top:top + h, left:left + w] = as_np(inp)
                    if not (c == as_np(out)).all():
                        ok = False
                        break
                if ok:
                    out.append((f"pad({top},{left})",
                                lambda a, t=top, l=left, h=h, w=w: as_list(
                                    np.pad(a, ((t, 0), (l, 0)))[:h, :w]
                                ) if False else
                                (lambda c: c)(
                                    np.pad(a, ((t, max(0, 0)), (l, 0))))))
    return out


def candidates_half_crop(pairs):
    out = []
    for name, sl in [("left-half", lambda a: a[:, :a.shape[1] // 2]),
                     ("right-half", lambda a: a[:, a.shape[1] // 2:]),
                     ("top-half", lambda a: a[:a.shape[0] // 2, :]),
                     ("bottom-half", lambda a: a[a.shape[0] // 2:, :])]:
        if all(sl(as_np(inp)).tolist() == out for inp, out in pairs):
            out.append((name, sl))
    return out


def candidates_mirror_concat(pairs):
    out = []
    for name, fn in [("hcat-mirror", lambda a: np.hstack([a, a[:, ::-1]])),
                     ("vcat-mirror", lambda a: np.vstack([a, a[::-1, :]])),
                     ("hcat-mirror-r", lambda a: np.hstack([a[:, ::-1], a])),
                     ("vcat-mirror-r", lambda a: np.vstack([a[::-1, :], a]))]:
        if all(fn(as_np(inp)).tolist() == out for inp, out in pairs):
            out.append((name, fn))
    return out


def candidates_gravity(pairs):
    """cell-wise gravity: every colored cell falls toward an edge."""
    out = []
    for name, dirn in [("grav-down", 0), ("grav-up", 1), ("grav-left", 2), ("grav-right", 3)]:
        ok = True
        for inp, out in pairs:
            a = as_np(inp)
            b = as_np(out)
            if a.shape != b.shape:
                ok = False
                break
            res = np.zeros_like(a)
            bg = Counter(a.ravel().tolist()).most_common(1)[0][0]
            if dirn == 0:  # down
                for j in range(a.shape[1]):
                    col = [x for x in a[:, j] if x != bg]
                    res[-len(col):, j] = col or 0
            elif dirn == 1:  # up
                for j in range(a.shape[1]):
                    col = [x for x in a[:, j] if x != bg]
                    res[:len(col), j] = col or 0
            elif dirn == 2:  # left
                for i in range(a.shape[0]):
                    row = [x for x in a[i, :] if x != bg]
                    res[i, :len(row)] = row or 0
            else:  # right
                for i in range(a.shape[0]):
                    row = [x for x in a[i, :] if x != bg]
                    res[i, -len(row):] = row or 0
            if not (res == b).all():
                ok = False
                break
        if ok:
            def f(a, dirn=dirn):
                a = a.copy()
                res = np.zeros_like(a)
                bg = Counter(a.ravel().tolist()).most_common(1)[0][0]
                if dirn == 0:
                    for j in range(a.shape[1]):
                        col = [x for x in a[:, j] if x != bg]
                        res[-len(col):, j] = col or 0
                elif dirn == 1:
                    for j in range(a.shape[1]):
                        col = [x for x in a[:, j] if x != bg]
                        res[:len(col), j] = col or 0
                elif dirn == 2:
                    for i in range(a.shape[0]):
                        row = [x for x in a[i, :] if x != bg]
                        res[i, :len(row)] = row or 0
                else:
                    for i in range(a.shape[0]):
                        row = [x for x in a[i, :] if x != bg]
                        res[i, -len(row):] = row or 0
                return res.tolist()
            out.append((name, f))
    return out


def candidates_line_connect(pairs):
    """fill h/v gaps between aligned same-color cells."""
    out = []
    for color in range(10):
        ok = True
        for inp, out in pairs:
            a = as_np(inp)
            b = as_np(out)
            if a.shape != b.shape:
                ok = False
                break
            res = a.copy()
            ys, xs = np.where(a == color)
            if len(ys) >= 2:
                # horizontal runs: same row, fill spans of zeros between endpoints
                for r in np.unique(ys):
                    cs = xs[ys == r]
                    if len(cs) >= 2:
                        for c0, c1 in zip(cs[:-1], cs[1:]):
                            span = a[r, c0:c1 + 1]
                            if set(np.unique(span).tolist()) <= {0, color}:
                                res[r, c0:c1 + 1] = color
                # vertical runs
                for ccol in np.unique(xs):
                    rs = ys[xs == ccol]
                    if len(rs) >= 2:
                        for r0, r1 in zip(rs[:-1], rs[1:]):
                            span = a[r0:r1 + 1, ccol]
                            if set(np.unique(span).tolist()) <= {0, color}:
                                res[r0:r1 + 1, ccol] = color
            if not (res == b).all():
                ok = False
                break
        if ok and any(len(np.where(as_np(inp) == color)[0]) >= 2 for inp, out in pairs):
            def f(a, color=color):
                a = a.copy()
                ys, xs = np.where(a == color)
                if len(ys) >= 2:
                    for r in np.unique(ys):
                        cs = xs[ys == r]
                        if len(cs) >= 2:
                            for c0, c1 in zip(cs[:-1], cs[1:]):
                                span = a[r, c0:c1 + 1]
                                if set(np.unique(span).tolist()) <= {0, color}:
                                    a[r, c0:c1 + 1] = color
                    for ccol in np.unique(xs):
                        rs = ys[xs == ccol]
                        if len(rs) >= 2:
                            for r0, r1 in zip(rs[:-1], rs[1:]):
                                span = a[r0:r1 + 1, ccol]
                                if set(np.unique(span).tolist()) <= {0, color}:
                                    a[r0:r1 + 1, ccol] = color
                return a.tolist()
            out.append((f"lines-c{color}", f))
    return out


def candidates_fill_enclosed(pairs):
    """bg cells fully enclosed by color c become c (flood fill from edges)."""
    out = []
    for color in range(1, 10):
        ok = True
        for inp, out in pairs:
            a = as_np(inp)
            b = as_np(out)
            if a.shape != b.shape:
                ok = False
                break
            if color not in a:
                ok = False
                break
            res = a.copy()
            # enclosed = bg not reachable from border without crossing color
            bg = 0
            h, w = a.shape
            visited = np.zeros((h, w), bool)
            stack = [(i, 0) for i in range(h)] + [(i, w - 1) for i in range(h)] + \
                    [(0, j) for j in range(w)] + [(h - 1, j) for j in range(w)]
            for i, j in stack:
                if a[i, j] == color:
                    continue
                visited[i, j] = True
            while stack:
                i, j = stack.pop()
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj] and a[ni, nj] != color:
                        visited[ni, nj] = True
                        stack.append((ni, nj))
            res[(~visited) & (a == bg)] = color
            if not (res == b).all():
                ok = False
                break
        if ok:
            def f(a, color=color):
                a = a.copy()
                h, w = a.shape
                visited = np.zeros((h, w), bool)
                stack = [(i, 0) for i in range(h)] + [(i, w - 1) for i in range(h)] + \
                        [(0, j) for j in range(w)] + [(h - 1, j) for j in range(w)]
                for i, j in stack:
                    if a[i, j] != color:
                        visited[i, j] = True
                while stack:
                    i, j = stack.pop()
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj] and a[ni, nj] != color:
                            visited[ni, nj] = True
                            stack.append((ni, nj))
                res = a.copy()
                res[(~visited) & (a == 0)] = color
                return res.tolist()
            out.append((f"enclosed-c{color}", f))
    return out


def candidates_remove_bg(pairs):
    """output = input with background set to 0 after recoloring (bg from output)."""
    for inp, out in pairs:
        a = as_np(inp)
        b = as_np(out)
        if a.shape != b.shape:
            return []
    # map: in-cell -> out-cell; identity unless output bg differs
    for inp0, out0 in pairs[:1]:
        a, b = as_np(inp0), as_np(out0)
        bg_out = Counter(b.ravel().tolist()).most_common(1)[0][0]
        res = a.copy()
        res[a == 0] = bg_out
        if not (res == b).all():
            continue
        if all((lambda c: (c == b).all())(
                (lambda t: np.where(t == 0, bg_out, t))(as_np(inp)))
               for inp, out in pairs):
            def f(a, bg_out=bg_out):
                t = a.copy()
                t[t == 0] = bg_out
                return t.tolist()
            return [("bg-replace", f)]
    return []


def candidates_object_bboxes(pairs):
    """output = each color's bbox corner markers? skip complex; simple: draw
    the bbox rectangle of each non-bg connected object in its color."""
    # too complex for v1 battery; placeholder
    return []


# --------------------------------------------------------------------------
# object-based rules (connected components)
# --------------------------------------------------------------------------

def connected_components(a, bg=0):
    """Label connected components (4-connectivity) of non-bg cells."""
    h, w = a.shape
    labels = np.zeros((h, w), dtype=np.int64)
    n = 0
    for i in range(h):
        for j in range(w):
            if a[i, j] == bg or labels[i, j] != 0:
                continue
            n += 1
            color = a[i, j]
            stack = [(i, j)]
            labels[i, j] = n
            while stack:
                x, y = stack.pop()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and labels[nx, ny] == 0 and a[nx, ny] == color:
                        labels[nx, ny] = n
                        stack.append((nx, ny))
    return labels, n


def candidates_object_gravity(pairs):
    """move each connected component as far as possible toward an edge."""
    out = []
    for name, dirn in [("obj-grav-down", 0), ("obj-grav-up", 1),
                       ("obj-grav-left", 2), ("obj-grav-right", 3)]:
        ok = True
        for inp, out in pairs:
            a = as_np(inp)
            b = as_np(out)
            if a.shape != b.shape:
                ok = False
                break
            labels, n = connected_components(a)
            h, w = a.shape
            res = np.zeros_like(a)
            for c in range(1, n + 1):
                ys, xs = np.where(labels == c)
                if dirn == 0:  # down: keep column, drop to bottom until blocked
                    for j in sorted(set(xs.tolist())):
                        cells = [y for y in sorted(ys[xs == j].tolist())]
                        # place cells at bottom of column j
                        for k, y in enumerate(cells):
                            res[h - len(cells) + k, j] = a[y, j]
                elif dirn == 1:
                    for j in sorted(set(xs.tolist())):
                        cells = [y for y in sorted(ys[xs == j].tolist())]
                        for k, y in enumerate(cells):
                            res[k, j] = a[y, j]
                elif dirn == 2:
                    for i in sorted(set(ys.tolist())):
                        cells = [x for x in sorted(xs[ys == i].tolist())]
                        for k, x in enumerate(cells):
                            res[i, k] = a[i, x]
                else:
                    for i in sorted(set(ys.tolist())):
                        cells = [x for x in sorted(xs[ys == i].tolist())]
                        for k, x in enumerate(cells):
                            res[i, w - len(cells) + k] = a[i, x]
            if not (res == b).all():
                ok = False
                break
        if ok and n > 0:
            def f(a, dirn=dirn):
                a = a.copy()
                labels, n = connected_components(a)
                h, w = a.shape
                res = np.zeros_like(a)
                for c in range(1, n + 1):
                    ys, xs = np.where(labels == c)
                    if dirn == 0:
                        for j in sorted(set(xs.tolist())):
                            cells = [y for y in sorted(ys[xs == j].tolist())]
                            for k, y in enumerate(cells):
                                res[h - len(cells) + k, j] = a[y, j]
                    elif dirn == 1:
                        for j in sorted(set(xs.tolist())):
                            cells = [y for y in sorted(ys[xs == j].tolist())]
                            for k, y in enumerate(cells):
                                res[k, j] = a[y, j]
                    elif dirn == 2:
                        for i in sorted(set(ys.tolist())):
                            cells = [x for x in sorted(xs[ys == i].tolist())]
                            for k, x in enumerate(cells):
                                res[i, k] = a[i, x]
                    else:
                        for i in sorted(set(ys.tolist())):
                            cells = [x for x in sorted(xs[ys == i].tolist())]
                            for k, x in enumerate(cells):
                                res[i, w - len(cells) + k] = a[i, x]
                return res.tolist()
            out.append((name, f))
    return out


def candidates_hollow(pairs):
    """cells fully surrounded (4-neigh) by same color become bg."""
    for inp, out in pairs:
        a = as_np(inp)
        if a.shape != as_np(out).shape:
            return []
    for inp0, out0 in pairs[:1]:
        a, b = as_np(inp0), as_np(out0)
        res = a.copy()
        h, w = a.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                c = a[i, j]
                if c != 0 and all(a[i + di, j + dj] == c for di, dj in
                                   ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    res[i, j] = 0
        if not (res == b).all():
            continue
        if all((lambda g: g == out)(
                (lambda t: _hollow(t))(as_np(inp)))
               for inp, out in pairs):
            return [("hollow", lambda g: _hollow(g).tolist())]
    return []


def _hollow(a):
    a = a.copy()
    h, w = a.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            c = a[i, j]
            if c != 0 and all(a[i + di, j + dj] == c for di, dj in
                              ((1, 0), (-1, 0), (0, 1), (0, -1))):
                a[i, j] = 0
    return a


def candidates_bbox_upscale(pairs):
    """output = upscale_k(bbox(input)) for some k in 2..6."""
    out = []
    for k in range(2, 7):
        ok = True
        for inp, out in pairs:
            a = as_np(inp)
            bg = Counter(a.ravel().tolist()).most_common(1)[0][0]
            nz = np.argwhere(a != bg)
            if len(nz) == 0:
                ok = False
                break
            y0, x0 = nz.min(0)
            y1, x1 = nz.max(0)
            bb = a[y0:y1 + 1, x0:x1 + 1]
            up = np.repeat(np.repeat(bb, k, axis=0), k, axis=1)
            if up.shape != as_np(out).shape or not (up == as_np(out)).all():
                ok = False
                break
        if ok:
            def f(a, k=k):
                bg = Counter(a.ravel().tolist()).most_common(1)[0][0]
                nz = np.argwhere(a != bg)
                if len(nz) == 0:
                    return a.tolist()
                y0, x0 = nz.min(0)
                y1, x1 = nz.max(0)
                bb = a[y0:y1 + 1, x0:x1 + 1]
                return np.repeat(np.repeat(bb, k, axis=0), k, axis=1).tolist()
            out.append((f"bbox-upscale{k}", f))
    return out


def candidates_upscale_bbox(pairs):
    """output = bbox(upscale_k(input)): scale whole grid then crop to fg bbox."""
    out = []
    for k in range(2, 7):
        ok = True
        for inp, out in pairs:
            a = as_np(inp)
            up = np.repeat(np.repeat(a, k, axis=0), k, axis=1)
            bg = Counter(up.ravel().tolist()).most_common(1)[0][0]
            nz = np.argwhere(up != bg)
            if len(nz) == 0:
                ok = False
                break
            y0, x0 = nz.min(0)
            y1, x1 = nz.max(0)
            bb = up[y0:y1 + 1, x0:x1 + 1]
            if bb.shape != as_np(out).shape or not (bb == as_np(out)).all():
                ok = False
                break
        if ok:
            def f(a, k=k):
                up = np.repeat(np.repeat(a, k, axis=0), k, axis=1)
                bg = Counter(up.ravel().tolist()).most_common(1)[0][0]
                nz = np.argwhere(up != bg)
                if len(nz) == 0:
                    return up.tolist()
                y0, x0 = nz.min(0)
                y1, x1 = nz.max(0)
                return up[y0:y1 + 1, x0:x1 + 1].tolist()
            out.append((f"upscale{k}-bbox", f))
    return out


def candidates_stretch(pairs):
    """output = input resized with nearest-neighbor sampling at a ratio
    consistent across all train pairs."""
    out = []
    ratios = []
    for inp, o in pairs:
        a = as_np(inp)
        if a.shape[0] == 0 or a.shape[1] == 0:
            return out
        ratios.append((len(o) / a.shape[0], len(o[0]) / a.shape[1]))
    if len(set(ratios)) != 1:
        return out
    rh, rw = ratios[0]
    if all(_stretch(as_np(inp), rh, rw).tolist() == o for inp, o in pairs):
        out.append((f"stretch({rh:.2f},{rw:.2f})",
                    lambda a, rh=rh, rw=rw: _stretch(a, rh, rw).tolist()))
    return out


def _stretch(a, rh, rw):
    h, w = a.shape
    oh, ow = int(round(h * rh)), int(round(w * rw))
    row_idx = np.minimum((np.arange(oh) * h / max(oh, 1)).astype(int), h - 1)
    col_idx = np.minimum((np.arange(ow) * w / max(ow, 1)).astype(int), w - 1)
    return a[np.ix_(row_idx, col_idx)]


def candidates_tile(pairs):
    """output = input tiled (np.tile) by integer factors."""
    out = []
    for inp0, out0 in pairs[:1]:
        a, b = as_np(inp0), as_np(out0)
        if a.shape[0] == 0 or a.shape[1] == 0:
            return out
        if b.shape[0] % a.shape[0] or b.shape[1] % a.shape[1]:
            return out
        thn, twn = b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]
        if thn == 1 and twn == 1:
            return out
        if all(np.tile(as_np(inp), (thn, twn)).tolist() == o for inp, o in pairs):
            out.append((f"tile({thn},{twn})",
                        lambda a, thn=thn, twn=twn: np.tile(a, (thn, twn)).tolist()))
    return out


def candidates_rect_fill(pairs):
    """for color c, fill the rectangle spanning all c-cells."""
    out = []
    for color in range(1, 10):
        ok = True
        for inp, o in pairs:
            a = as_np(inp)
            b = as_np(o)
            if a.shape != b.shape or color not in a:
                ok = False
                break
            ys, xs = np.where(a == color)
            if len(ys) < 2:
                ok = False
                break
            span = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            if not set(np.unique(span).tolist()) <= {0, color}:
                ok = False
                break
            res = a.copy()
            res[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = color
            if not (res == b).all():
                ok = False
                break
        if ok:
            def f(a, color=color):
                a = a.copy()
                ys, xs = np.where(a == color)
                if len(ys) >= 2:
                    a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = color
                return a.tolist()
            out.append((f"rectfill-c{color}", f))
    return out


def candidates_erase(pairs):
    """cells of color c become background."""
    out = []
    for color in range(1, 10):
        if all(
            np.where(as_np(inp) == color, 0, as_np(inp)).tolist() == o
            for inp, o in pairs
        ) and any(color in np.unique(as_np(inp)) for inp, o in pairs):
            def f(a, color=color):
                t = a.copy()
                t[t == color] = 0
                return t.tolist()
            out.append((f"erase-c{color}", f))
    return out


def candidates_border(pairs):
    """add a 1-cell border of color c around the grid."""
    out = []
    for color in range(1, 10):
        if all(np.pad(as_np(inp), 1, constant_values=color).tolist() == o
               for inp, o in pairs):
            def f(a, color=color):
                return np.pad(a, 1, constant_values=color).tolist()
            out.append((f"border-c{color}", f))
    return out


def candidates_count_canvas(pairs):
    """output = 1xk or kx1 canvas of color c, k = count of c-cells or c-objects."""
    out = []
    for color in range(1, 10):
        for mode in ("cells", "objects"):
            ok = True
            for inp, o in pairs:
                a = as_np(inp)
                if mode == "cells":
                    k = int((a == color).sum())
                else:
                    labels, n = connected_components(a)
                    k = sum(1 for c in range(1, n + 1)
                            if a[labels == c][0] == color)
                b = as_np(o)
                if not ((b.shape == (1, k) and (b == color).all()) or
                        (b.shape == (k, 1) and (b == color).all())):
                    ok = False
                    break
            if ok:
                a0, o0 = pairs[0]
                b0 = as_np(o0)
                horiz = b0.shape[0] == 1
                def f(a, color=color, mode=mode, horiz=horiz):
                    if mode == "cells":
                        k = int((a == color).sum())
                    else:
                        labels, n = connected_components(a)
                        k = sum(1 for c in range(1, n + 1)
                                if a[labels == c][0] == color)
                    k = max(k, 1)
                    return [[color] * k] if horiz else [[color] for _ in range(k)]
                out.append((f"count-{mode}-c{color}", f))
    return out


ALL_RULE_FNS = [
    ("geometry", candidates_geometry),
    ("color-map", candidates_color_map),
    ("upscale", candidates_upscale),
    ("downscale", candidates_downscale),
    ("stretch", candidates_stretch),
    ("tile", candidates_tile),
    ("crop", candidates_crop),
    ("bbox", candidates_bbox),
    ("half-crop", candidates_half_crop),
    ("mirror-concat", candidates_mirror_concat),
    ("gravity", candidates_gravity),
    ("obj-gravity", candidates_object_gravity),
    ("line-connect", candidates_line_connect),
    ("rect-fill", candidates_rect_fill),
    ("fill-enclosed", candidates_fill_enclosed),
    ("bg-replace", candidates_remove_bg),
    ("erase", candidates_erase),
    ("border", candidates_border),
    ("count-canvas", candidates_count_canvas),
    ("hollow", candidates_hollow),
    ("bbox-upscale", candidates_bbox_upscale),
    ("upscale-bbox", candidates_upscale_bbox),
]


def solve_task(train_pairs, test_input, with_tags=False):
    """Return ordered list of candidate test-output grids."""
    cands: list[tuple[str, object]] = []
    for group_name, fn in ALL_RULE_FNS:
        try:
            for tag, tf in fn(train_pairs):
                cands.append((tag, tf))
        except Exception:
            continue
    results = []
    seen = set()
    for tag, tf in cands:
        try:
            g = tf(as_np(test_input))
            gl = as_list(g) if isinstance(g, np.ndarray) else g
            key = str(gl)
            if key not in seen:
                seen.add(key)
                results.append((tag, gl) if with_tags else gl)
        except Exception:
            continue
    return results


def predict_output_size(train_pairs, test_input):
    """Predict (h, w) of test output from train size ratios."""
    ratios = []
    for inp, out in train_pairs:
        ih, iw = len(inp), len(inp[0])
        oh, ow = len(out), len(out[0])
        ratios.append((oh / ih, ow / iw))
    # exact ratio consistency check
    exact = None
    from collections import Counter as C
    rc = C(ratios)
    (rh, rw), _ = rc.most_common(1)[0]
    if len(rc) == 1:
        th, tw = len(test_input), len(test_input[0])
        return (int(round(th * rh)), int(round(tw * rw)))
    # heuristic: same size if any pair same-size and majority same
    same = sum(1 for r in ratios if r == (1.0, 1.0))
    if same >= len(ratios) // 2:
        return (len(test_input), len(test_input[0]))
    return (len(test_input), len(test_input[0]))


def make_attempts(task, test_input):
    """Produce attempt_1 and attempt_2 for a task's test input."""
    train_pairs = [(p["input"], p["output"]) for p in task["train"]]
    cands = solve_task(train_pairs, test_input)
    ph, pw = predict_output_size(train_pairs, test_input)
    in_np = as_np(test_input)
    ih, iw = in_np.shape

    def fallback_same_size():
        # color-mapped copy if consistent map exists, else plain copy
        cm = learn_color_map(train_pairs)
        if cm and all(len(inp) == len(out) and len(inp[0]) == len(out[0])
                      for inp, out in train_pairs):
            return as_list(np.vectorize(lambda x: cm.get(int(x), int(x)))(in_np))
        return test_input

    def fallback_resize():
        if ph <= 0 or pw <= 0:
            return test_input
        if (ph, pw) == (ih, iw):
            return fallback_same_size()
        # exact integer downscale via block collapse when divisibility holds
        if ph < ih and pw < iw and ih % ph == 0 and iw % pw == 0:
            kh, kw = ih // ph, iw // pw
            return as_list(in_np.reshape(ph, kh, pw, kw)[:, 0, :, 0])
        # nearest-neighbor resampling (works for any target size)
        row_idx = np.minimum((np.arange(ph) * ih / ph).astype(int), ih - 1)
        col_idx = np.minimum((np.arange(pw) * iw / pw).astype(int), iw - 1)
        return as_list(in_np[np.ix_(row_idx, col_idx)])

    a1 = a2 = None
    if len(cands) >= 1:
        a1 = cands[0]
    if len(cands) >= 2:
        a2 = cands[1]
    elif len(cands) == 1:
        # decorrelated second shot: predicted-size content guess
        r = fallback_resize()
        a2 = r if (len(r), len(r[0])) != (len(a1), len(a1[0])) else fallback_same_size()
    if a1 is None:
        a1 = fallback_resize()
    if a2 is None:
        a2 = fallback_same_size()
        if a2 == a1:
            # second distinct fallback: canvas filled with most frequent train-output color
            cols = Counter()
            for _, out in train_pairs:
                for row in out:
                    cols.update(row)
            if cols:
                top = cols.most_common(1)[0][0]
                if (ph, pw) == (ih, iw):
                    a2 = [[top] * pw for _ in range(ph)]
                else:
                    a2 = [[0] * pw for _ in range(ph)]
                    for i, row in enumerate(test_input):
                        for j, v in enumerate(row):
                            if i < ph and j < pw:
                                a2[i][j] = v
    # final safety net: never emit None or empty grids (Kaggle rejects them)
    if a1 is None or not a1 or not a1[0]:
        a1 = test_input
    if a2 is None or not a2 or not a2[0]:
        a2 = test_input
    return a1, a2
