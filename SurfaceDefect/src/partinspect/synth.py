"""Synthetic defects for moulded plastic parts, with exact masks.

Every generator changes pixels **only inside the mask it returns**, so the mask is an exact label and the rest
of the image is untouched (tested in tests/test_synth.py). Positions of the two features that the original
photoshopped NG images removed (round mark, engraved text) are given for the 1163x803 photos of this part;
they are scaled to other image sizes.

Types:
    missing_mark   the round mark on the right side is gone (filled with plastic texture)
    erased_text    part of the engraved text on the right side is gone
    scratch        thin bright or dark line on the part
    spot           dark contamination dots
    edge_chip      a piece missing at the edge of the part (short shot / broken edge)
    burn           discoloured (darker, brownish) area
"""
import cv2
import numpy as np

REF_SIZE = (1163, 803)  # (w, h) of the reference photos
MARK = (1106, 557, 42)  # centre x, centre y, radius of the right round mark
TEXT = (1077, 190, 1133, 505)  # x0, y0, x1, y1 of the right engraved text band
PLAIN = (15, 305, 95, 495)  # x0, y0, x1, y1 of a plain stretch of the left bar (texture donor, no engraving)
TYPES = ["missing_mark", "erased_text", "scratch", "spot", "edge_chip", "burn"]


def foreground(img, thr=200):
    """Part pixels (the background is near-white)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < thr


def _scale(img):
    h, w = img.shape[:2]
    return w / REF_SIZE[0], h / REF_SIZE[1]


def _texture_fill(img, mask, rng, feather=4):
    """Fill mask with real plastic texture copied from the plain stretch of the left bar (PLAIN).

    The bounding box of the mask is covered with tiles as large as PLAIN allows, each from a random offset
    inside PLAIN, so no engraving or mark is copied in. The copy is blended with the original over the first
    `feather` pixels inside the mask: the seam is soft and nothing outside the mask changes.
    """
    sx, sy = _scale(img)
    px0, py0, px1, py1 = int(PLAIN[0] * sx), int(PLAIN[1] * sy), int(PLAIN[2] * sx), int(PLAIN[3] * sy)
    yy, xx = np.where(mask)
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    th, tw = min(y1 - y0, py1 - py0), min(x1 - x0, px1 - px0)
    out = img.copy()
    for ty in range(y0, y1, th):
        for tx in range(x0, x1, tw):
            h, w = min(th, y1 - ty), min(tw, x1 - tx)
            oy, ox = rng.integers(py0, py1 - h + 1), rng.integers(px0, px1 - w + 1)
            out[ty:ty + h, tx:tx + w] = img[oy:oy + h, ox:ox + w]
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(dist / feather, 0, 1)[..., None]
    return (alpha * out + (1 - alpha) * img).astype(np.uint8)


def _random_fg_point(img, rng, margin=40):
    ys, xs = np.where(foreground(img))
    h, w = img.shape[:2]
    ok = (xs > margin) & (xs < w - margin) & (ys > margin) & (ys < h - margin)
    i = rng.choice(np.flatnonzero(ok))
    return int(xs[i]), int(ys[i])


def missing_mark(img, rng):
    sx, sy = _scale(img)
    cx, cy, r = MARK
    cx, cy = int((cx + rng.integers(-6, 7)) * sx), int((cy + rng.integers(-6, 7)) * sy)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(mask, (cx, cy), (int(r * sx * rng.uniform(0.9, 1.1)), int(r * sy * rng.uniform(0.9, 1.1))),
                0, 0, 360, 1, -1)
    return mask.astype(bool)


def erased_text(img, rng):
    sx, sy = _scale(img)
    x0, y0, x1, y1 = TEXT
    length = rng.integers(80, y1 - y0)
    start = rng.integers(y0, y1 - length + 1)
    mask = np.zeros(img.shape[:2], bool)
    mask[int(start * sy):int((start + length) * sy), int(x0 * sx):int(x1 * sx)] = True
    return mask & cv2.dilate(foreground(img).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)


def scratch(img, rng):
    x, y = _random_fg_point(img, rng)
    pts = [(x, y)]
    angle = rng.uniform(0, np.pi)
    for _ in range(rng.integers(2, 5)):
        angle += rng.normal(0, 0.3)
        step = rng.integers(15, 50)
        x, y = int(x + step * np.cos(angle)), int(y + step * np.sin(angle))
        pts.append((x, y))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.polylines(mask, [np.array(pts, np.int32)], False, 1, int(rng.integers(2, 5)))
    return mask.astype(bool) & foreground(img)


def spot(img, rng):
    mask = np.zeros(img.shape[:2], np.uint8)
    x, y = _random_fg_point(img, rng)
    for _ in range(rng.integers(1, 4)):
        c = (int(x + rng.integers(-25, 26)), int(y + rng.integers(-25, 26)))
        cv2.ellipse(mask, c, (int(rng.integers(4, 11)), int(rng.integers(4, 11))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
    return mask.astype(bool) & foreground(img)


def edge_chip(img, rng):
    fg = foreground(img).astype(np.uint8)
    edge = fg - cv2.erode(fg, np.ones((3, 3), np.uint8))
    ys, xs = np.where(edge > 0)
    h, w = fg.shape
    ok = (xs > 40) & (xs < w - 40) & (ys > 40) & (ys < h - 40)
    i = rng.choice(np.flatnonzero(ok))
    n = rng.integers(5, 9)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = rng.uniform(18, 40, n)
    poly = np.stack([xs[i] + rad * np.cos(ang), ys[i] + rad * np.sin(ang)], 1).astype(np.int32)
    mask = np.zeros_like(fg)
    cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool) & fg.astype(bool)


def burn(img, rng):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(mask, _random_fg_point(img, rng), (int(rng.integers(12, 35)), int(rng.integers(12, 35))),
                float(rng.uniform(0, 180)), 0, 360, 1, -1)
    return mask.astype(bool) & foreground(img)


def apply(img, kind, rng):
    """Return (defective image, bool mask). Pixels outside the mask are bit-identical to `img`."""
    mask = globals()[kind](img, rng)
    if kind in ("missing_mark", "erased_text"):
        out = _texture_fill(img, mask, rng)
    elif kind == "scratch":
        delta = rng.choice([-1, 1]) * rng.uniform(45, 90)
        out = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    elif kind == "spot":
        out = (img.astype(np.float32) * rng.uniform(0.15, 0.4)).astype(np.uint8)
    elif kind == "edge_chip":
        bg = img[~foreground(img)]
        out = np.clip(rng.normal(bg.mean(0), bg.std(0) + 1, img.shape), 0, 255).astype(np.uint8)
    else:  # burn: darker and shifted towards brown
        out = np.clip(img.astype(np.float32) * np.array([0.45, 0.55, 0.75]) + np.array([0, 10, 30]), 0, 255).astype(np.uint8)
    result = img.copy()
    result[mask] = out[mask]
    return result, mask
