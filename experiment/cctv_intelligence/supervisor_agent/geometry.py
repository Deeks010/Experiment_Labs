"""Geometry and spatial relations in camera pixels.

All distances are reported in body heights (bh): pixel distance divided by the person's
bounding-box height at that spot. This keeps relations meaningful under perspective
without floor calibration. 1 bh is about 1.7 m.
"""
from __future__ import annotations

import math


def point_in_poly(x: float, y: float, pts) -> bool:
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]; xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def poly_dist(x: float, y: float, pts) -> float:
    """0 inside, else distance to the nearest edge (pixels)."""
    if point_in_poly(x, y, pts):
        return 0.0
    n = len(pts)
    return min(_seg_dist(x, y, *pts[i], *pts[(i + 1) % n]) for i in range(n))


def zones_at(x, y, zones, kind=None) -> list[dict]:
    return [z for z in zones if (kind is None or z["kind"] == kind) and point_in_poly(x, y, z["points"])]


def side_of(x, y, z) -> tuple[str, float | None]:
    """Image-relative side of a point versus an object's bounding box.
    'far side' = above the object in the image (usually away from the camera),
    'camera side' = below it. Returns (side, fraction along the object's width or None)."""
    x1, y1, x2, y2 = z["box"]
    along = (x - x1) / ((x2 - x1) or 1)
    if x1 <= x <= x2:
        if y < y1:
            return "far side", along
        if y > y2:
            return "camera side", along
        return "over/at", along
    vert = "far side" if y < y1 else ("camera side" if y > y2 else "level")
    end = "beyond LEFT end" if x < x1 else "beyond RIGHT end"
    return (end if vert == "level" else f"{end}, {vert}"), None


def part_word(along: float | None) -> str:
    if along is None:
        return ""
    return "left part" if along < 1 / 3 else ("middle" if along < 2 / 3 else "right part")


def relation_text(x, y, body_h, zones, n_objects=2) -> str:
    """Readable relation of a foot point to the map. Example:
    'in Op 15 | far side of Bench L (right part), 0.3 bh (~0.5 m); Press 15 0.4 bh (~0.7 m)'"""
    bh = max(body_h or 1.0, 1.0)
    inside = [z["name"] for z in zones_at(x, y, zones, "floor_area")]
    objs = sorted(((poly_dist(x, y, z["points"]) / bh, z) for z in zones if z["kind"] == "static_object"),
                  key=lambda t: t[0])[:n_objects]
    parts = []
    for d, z in objs:
        side, along = side_of(x, y, z)
        pw = part_word(along)
        rel = f"{side} of {z['name']}" + (f" ({pw})" if pw else "")
        parts.append(f"{rel}, {d:.1f} bh (~{d * 1.7:.1f} m)" if d > 0 else f"on/over {z['name']} footprint ({pw})")
    head = ("in " + " & ".join(inside)) if inside else "in no mapped floor area"
    between = ""
    if len(objs) == 2 and objs[0][0] < 1.2 and objs[1][0] < 1.2:
        s0, s1 = side_of(x, y, objs[0][1])[0], side_of(x, y, objs[1][1])[0]
        if {s0.split(",")[0], s1.split(",")[0]} == {"far side", "camera side"}:
            between = f" | BETWEEN {objs[0][1]['name']} and {objs[1][1]['name']}"
    return head + between + " | " + "; ".join(parts)


def place_label(x, y, zones) -> str:
    """Short place name used in movement episodes and zone RLE: the floor area, else the nearest object side."""
    inside = [z["name"] for z in zones_at(x, y, zones, "floor_area")]
    if inside:
        return inside[0]
    objs = [z for z in zones if z["kind"] == "static_object"]
    if not objs:
        return "unmapped floor"
    z = min(objs, key=lambda z: poly_dist(x, y, z["points"]))
    side, _ = side_of(x, y, z)
    if side == "over/at":
        return f"at {z['name']} (foot point over its footprint)"
    return f"unmapped floor ({side} of {z['name']})"


def edge_of(x, y, w, h, margin_frac, zones) -> str | None:
    """Name of the entry/exit edge for a point: an 'exit' zone name, else an image border."""
    for z in zones:
        if z.get("role") == "exit" and point_in_poly(x, y, z["points"]):
            return z["name"]
    mx, my = w * margin_frac, h * margin_frac
    if x < mx:
        return "left image edge"
    if x > w - mx:
        return "right image edge"
    if y > h - my:
        return "bottom image edge"
    if y < my:
        return "top image edge"
    return None
