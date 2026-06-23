import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image


OUTPUT_DIR = Path(__file__).resolve().parent
load_dotenv(OUTPUT_DIR / ".env")
DEBUG_CROPS_DIR = OUTPUT_DIR / "debug_crops"

GLOBAL_ALLOWED_CATEGORIES = {
    "machine",
    "workstation_table",
    "control_panel",
    "storage_rack",
    "safety_line",
    "readable_text_region",
    "fixed_signage",
}

FORBIDDEN_CATEGORY_WORDS = {
    "floor",
    "ceiling",
    "roof",
    "wall",
    "background",
    "empty_space",
    "person",
    "human",
    "shadow",
}

CHILD_WHITELISTS = {
    "machine": [
        "name_board",
        "logo",
        "local_control_panel",
        "gauge",
        "button_cluster",
        "indicator_light",
        "display",
    ],
    "workstation_table": [
        "crate",
        "tray",
        "scale",
        "document",
        "bin",
        "tool",
        "fixture",
    ],
    "control_panel": [
        "display",
        "button_cluster",
        "indicator_light",
        "label",
        "gauge",
        "switch",
    ],
    "storage_rack": ["crate", "bin", "tray", "container"],
}

CATEGORY_COLORS = {
    "machine": (0, 165, 255),
    "workstation_table": (255, 0, 255),
    "control_panel": (0, 255, 0),
    "storage_rack": (255, 0, 0),
    "large_bin": (255, 0, 0),
    "safety_line": (0, 0, 255),
    "readable_text_region": (0, 255, 255),
    "fixed_signage": (0, 255, 255),
    "fixed_fixture": (255, 255, 0),
    "name_board": (0, 255, 255),
    "logo": (0, 255, 255),
    "local_control_panel": (0, 255, 0),
    "gauge": (0, 255, 0),
    "button_cluster": (0, 255, 0),
    "indicator_light": (0, 255, 0),
    "display": (0, 255, 0),
    "crate": (255, 0, 0),
    "tray": (255, 0, 0),
    "scale": (0, 255, 0),
    "document": (255, 255, 255),
    "bin": (255, 0, 0),
    "label": (0, 255, 255),
}


def extract_frame(video_path, frame_idx=30):
    """Extract a specific frame from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_idx} from video")
    return frame


def load_frame(input_path, frame_idx=30):
    """Load either an image or a video frame."""
    path = Path(input_path)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.suffix.lower() in image_exts:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read image {input_path}")
        return frame
    return extract_frame(path, frame_idx=frame_idx)


def frame_to_pil(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Set it in experiment/cctv_intelligence/.env")
    return genai.Client(api_key=api_key)


def parse_json_response(response_text):
    """Parse JSON even if the model wraps it in a markdown fence."""
    text = response_text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


def generate_json(client, image, prompt):
    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    response = client.models.generate_content(
        model=model,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    return parse_json_response(response.text)


def get_scene_profile_gemini(client, frame):
    """Understand the CCTV scene and decide what belongs in its permanent map."""
    prompt = """
You are designing the first-stage object policy for a CCTV spatial calibration map.

Goal:
Understand what kind of place this camera is watching, then decide which visible things should be part of a permanent scene map.

Definition of permanent map object:
An object should be included only if it is likely to remain in the same place across normal CCTV footage and is useful as a spatial reference for later activity/incident analysis.

Include examples:
- fixed machines and fixed machine panels
- fixed workstation tables or benches
- fixed control panels
- mounted screens, gauges, button panels, indicator panels
- fixed racks, cages, cabinets, and bolted material stands
- fixed safety markings, lanes, barriers, signs, name boards
- one grouped fixed signage board or document holder, not each paper separately
- every visible repeated fixed machine or fixed station, as separate objects

Exclude examples:
- people or body parts
- loose tools, parts, product pieces, scrap, contents inside bins
- loose papers/documents unless they are on a fixed signage board or permanent document holder
- movable crates, totes, bins, carts, trays, and containers visible in only one frame
- floor, wall, ceiling, roof, shadows, glare, empty space
- ceiling lights, overhead ducts, roof beams, and generic building fixtures unless the feed is specifically about those assets
- generic blue frame/support structure when a more meaningful object like a table or machine can represent it

Do not return boxes in this step. Think like a site surveyor deciding the detection policy, not like an object detector.
Be conservative: if a visible item could reasonably be moved during normal operations, exclude it from the permanent map policy.

Return this exact JSON shape:
{
  "scene_profile": {
    "place_type": "short description of the place",
    "primary_workflow": "what work appears to happen here",
    "permanence_rule": "one sentence rule for what belongs in the permanent map"
  },
  "include_policy": [
    {
      "category": "one of: machine, workstation_table, control_panel, storage_rack, safety_line, readable_text_region, fixed_signage",
      "include_when": "specific visual/permanence condition",
      "exclude_when": "specific condition that makes it transient or not useful",
      "priority": "high|medium|low"
    }
  ],
  "exclude_policy": [
    "specific visual classes or conditions to ignore in this scene"
  ]
}
"""
    print("Stage 0: requesting scene profile and permanent-map policy from Gemini...")
    return generate_json(client, frame_to_pil(frame), prompt)


def format_scene_policy(scene_policy):
    return json.dumps(scene_policy, indent=2)


def get_global_anchors_gemini(client, frame, scene_policy):
    """Find only permanent operational anchors in the full CCTV frame."""
    prompt = f"""
You are creating stage 1 of a CCTV spatial calibration map.

First apply this scene-specific permanent-map policy:
{format_scene_policy(scene_policy)}

Now detect only visible objects that satisfy that policy.

Allowed output categories:
machine, workstation_table, control_panel, storage_rack, safety_line, readable_text_region, fixed_signage.

Permanent-map test:
- Include an object only if it is likely to remain in the same place across normal footage.
- Include it only if it helps locate later events, activities, persons, violations, or incidents.
- If uncertain whether an object is permanent or transient, exclude it from stage 1.
- Do not include movable containers, crates, totes, trays, carts, loose parts, work-in-progress, or material contents in stage 1.

Box rules:
- Draw one tight box around each included permanent object.
- Do not return parent assemblies when a clearer stable object exists. For example, prefer workstation_table over a whole workstation frame.
- Repeated fixed machines or fixed workstations are important anchors, not clutter. Return each visible instance separately.
- Never merge multiple machines into one box. Never skip a partially visible fixed machine if enough of it is visible to identify it.
- Do not return small child details in this pass unless they are fixed machine/station labels.
- For signage/SOP documents, return one grouped fixed_signage box around the mounted board or holder, not each document page.
- Do not return floor, wall, ceiling, roof, ceiling lights, overhead ducts, beams, shadows, glare, empty space, people, loose material, product contents, or temporary crates/totes.
- Use [ymin, xmin, ymax, xmax] scaled 0-1000.

Return this exact JSON shape:
{{
  "anchors": [
    {{
      "name": "short stable name",
      "category": "one allowed category",
      "permanence": "why this belongs in the permanent map",
      "color": "dominant color",
      "description": "brief visual description, include readable text if any",
      "box_2d": [0, 0, 0, 0],
      "confidence": 0.0,
      "children_expected": ["expected child categories"]
    }}
  ]
}}
"""
    print("Stage 1: requesting permanent global anchors from Gemini...")
    return generate_json(client, frame_to_pil(frame), prompt).get("anchors", [])


def get_roi_children_gemini(client, crop, anchor, allowed_categories):
    """Detect child objects within a cropped anchor region."""
    allowed_text = ", ".join(allowed_categories)
    prompt = f"""
You are analyzing a cropped region from a factory CCTV frame.

The crop contains one parent object:
- parent name: {anchor.get("name")}
- parent category: {anchor.get("category")}
- parent description: {anchor.get("description", "")}

Return only child objects from this whitelist:
{allowed_text}

Rules:
- Use coordinates relative to this crop, scaled 0-1000.
- Do not return the parent object itself.
- Do not return floor, wall, ceiling, roof, shadows, people, empty space, or support structures.
- Draw tight boxes around the visible child object only.
- For text/name boards/logos, draw the box around the physical label or printed region, not the whole machine face.
- Prefer missing an uncertain tiny object over drawing a large approximate box.

Return this exact JSON shape:
{{
  "items": [
    {{
      "name": "short child name",
      "category": "one whitelist category",
      "color": "dominant color",
      "description": "brief visual description, include readable text if any",
      "box_2d": [0, 0, 0, 0],
      "confidence": 0.0
    }}
  ]
}}
"""
    print(f"Stage 2: requesting ROI children for {anchor.get('id')} {anchor.get('name')}...")
    return generate_json(client, frame_to_pil(crop), prompt).get("items", [])


def clamp(value, low=0, high=1000):
    return max(low, min(high, int(round(value))))


def normalize_box(box):
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [clamp(float(value)) for value in box]
    except (TypeError, ValueError):
        return None
    if ymax <= ymin or xmax <= xmin:
        return None
    return [ymin, xmin, ymax, xmax]


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def vertical_overlap_ratio(box_a, box_b):
    overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    shortest = max(1, min(box_a[2] - box_a[0], box_b[2] - box_b[0]))
    return overlap / shortest


def horizontal_gap(box_a, box_b):
    if box_a[3] < box_b[1]:
        return box_b[1] - box_a[3]
    if box_b[3] < box_a[1]:
        return box_a[1] - box_b[3]
    return 0


def merge_boxes(boxes):
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def merge_fixed_signage_groups(items):
    """Collapse repeated document/page detections into permanent signage holders."""
    signage = [item for item in items if item.get("category") == "fixed_signage"]
    if len(signage) <= 1:
        return items

    others = [item for item in items if item.get("category") != "fixed_signage"]
    groups = []

    for item in sorted(signage, key=lambda value: (value["box_2d"][0], value["box_2d"][1])):
        item_box = item["box_2d"]
        matched_group = None
        for group in groups:
            group_box = merge_boxes([member["box_2d"] for member in group])
            if vertical_overlap_ratio(item_box, group_box) >= 0.25 and horizontal_gap(item_box, group_box) <= 120:
                matched_group = group
                break
        if matched_group is None:
            groups.append([item])
        else:
            matched_group.append(item)

    merged = []
    for index, group in enumerate(groups, start=1):
        if len(group) == 1:
            merged.append(group[0])
            continue
        merged_box = merge_boxes([member["box_2d"] for member in group])
        merged.append(
            {
                "id": None,
                "parent_id": None,
                "name": f"Fixed Signage Group {index}",
                "category": "fixed_signage",
                "color": ", ".join(sorted({member.get("color", "") for member in group if member.get("color")})),
                "description": f"Grouped permanent signage/document holder containing {len(group)} detected pages or panels.",
                "box_2d": merged_box,
                "confidence": max(member.get("confidence") or 0 for member in group),
                "source_stage": "global",
            }
        )

    return others + merged


def has_forbidden_category(item):
    category = str(item.get("category", "")).strip().lower()
    name = str(item.get("name", "")).strip().lower()
    return any(word in category or word in name for word in FORBIDDEN_CATEGORY_WORDS)


def clean_category(category):
    return str(category or "").strip().lower().replace(" ", "_")


def normalize_items(items, source_stage, allowed_categories=None, parent_id=None):
    normalized = []
    allowed = set(allowed_categories or [])

    for item in items:
        category = clean_category(item.get("category"))
        item["category"] = category

        if allowed and category not in allowed:
            continue
        if has_forbidden_category(item):
            continue

        box = normalize_box(item.get("box_2d"))
        if not box:
            continue

        area = box_area(box)
        if source_stage == "roi" and area > 450_000:
            continue
        if source_stage == "global" and category not in {"safety_line"} and area > 650_000:
            continue

        normalized.append(
            {
                "id": item.get("id"),
                "parent_id": parent_id,
                "name": str(item.get("name", "Unknown")).strip() or "Unknown",
                "category": category,
                "color": str(item.get("color", "")).strip(),
                "description": str(item.get("description", "")).strip(),
                "box_2d": box,
                "confidence": item.get("confidence"),
                "source_stage": source_stage,
            }
        )

    return normalized


def scaled_box_to_pixels(box, frame_shape):
    h, w = frame_shape[:2]
    ymin, xmin, ymax, xmax = box
    x1 = int((xmin / 1000.0) * w)
    y1 = int((ymin / 1000.0) * h)
    x2 = int((xmax / 1000.0) * w)
    y2 = int((ymax / 1000.0) * h)
    return x1, y1, x2, y2


def pixels_to_scaled_box(x1, y1, x2, y2, frame_shape):
    h, w = frame_shape[:2]
    return [
        clamp((y1 / h) * 1000),
        clamp((x1 / w) * 1000),
        clamp((y2 / h) * 1000),
        clamp((x2 / w) * 1000),
    ]


def crop_anchor(frame, box_2d, padding=0.04):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = scaled_box_to_pixels(box_2d, frame.shape)
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def translate_crop_box_to_global(child_box, crop_bounds, frame_shape):
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    crop_h = crop_y2 - crop_y1
    crop_w = crop_x2 - crop_x1
    y1 = crop_y1 + int((child_box[0] / 1000.0) * crop_h)
    x1 = crop_x1 + int((child_box[1] / 1000.0) * crop_w)
    y2 = crop_y1 + int((child_box[2] / 1000.0) * crop_h)
    x2 = crop_x1 + int((child_box[3] / 1000.0) * crop_w)
    return pixels_to_scaled_box(x1, y1, x2, y2, frame_shape)


def assign_ids(items, prefix):
    for index, item in enumerate(items, start=1):
        item["id"] = f"{prefix}_{index:03d}"
    return items


def draw_boxes(frame, mapping_data, output_path):
    """Draw compact numbered boxes plus a legend to avoid label overlap."""
    annotated = frame.copy()
    items = mapping_data.get("items", [])
    legend_rows = []

    for index, item in enumerate(items, start=1):
        category = item.get("category", "")
        box = item.get("box_2d")
        if not box:
            continue

        x1, y1, x2, y2 = scaled_box_to_pixels(box, annotated.shape)
        color = CATEGORY_COLORS.get(category, (0, 255, 0))
        thickness = 3 if item.get("source_stage") == "global" else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        badge = str(index)
        (text_w, text_h), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 8, y1), color, -1)
        cv2.putText(
            annotated,
            badge,
            (x1 + 4, max(text_h, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )
        legend_rows.append(f"{index}. {item.get('name')} [{category}/{item.get('source_stage')}]")

    legend_path = Path(output_path).with_suffix(".legend.txt")
    legend_path.write_text("\n".join(legend_rows), encoding="utf-8")
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved annotated image to {output_path}")
    print(f"Saved annotation legend to {legend_path}")


def run_staged_mapping(frame, max_roi_calls=None):
    client = get_gemini_client()

    scene_policy = get_scene_profile_gemini(client, frame)
    policy_path = OUTPUT_DIR / "scene_profile.json"
    policy_path.write_text(json.dumps(scene_policy, indent=2), encoding="utf-8")
    print(f"Saved scene profile to {policy_path}")

    raw_anchors = get_global_anchors_gemini(client, frame, scene_policy)
    anchors = normalize_items(raw_anchors, "global", GLOBAL_ALLOWED_CATEGORIES)
    anchors = merge_fixed_signage_groups(anchors)
    assign_ids(anchors, "anchor")

    DEBUG_CROPS_DIR.mkdir(exist_ok=True)
    all_items = list(anchors)
    roi_calls = 0

    for anchor in anchors:
        allowed_children = CHILD_WHITELISTS.get(anchor["category"])
        if not allowed_children:
            continue
        if max_roi_calls is not None and roi_calls >= max_roi_calls:
            break

        crop, crop_bounds = crop_anchor(frame, anchor["box_2d"])
        crop_path = DEBUG_CROPS_DIR / f"{anchor['id']}_{anchor['category']}.jpg"
        cv2.imwrite(str(crop_path), crop)

        raw_children = get_roi_children_gemini(client, crop, anchor, allowed_children)
        children = normalize_items(raw_children, "roi", allowed_children, parent_id=anchor["id"])

        for child_index, child in enumerate(children, start=1):
            child["id"] = f"{anchor['id']}_child_{child_index:03d}"
            child["box_2d"] = translate_crop_box_to_global(child["box_2d"], crop_bounds, frame.shape)

        all_items.extend(children)
        roi_calls += 1

    return {
        "metadata": {
            "pipeline": "scene_policy_plus_global_anchor_plus_roi_children",
            "coordinate_format": "[ymin, xmin, ymax, xmax] scaled 0-1000",
            "global_anchor_count": len(anchors),
            "roi_call_count": roi_calls,
            "item_count": len(all_items),
        },
        "scene_policy": scene_policy,
        "items": all_items,
    }


def main(input_path, frame_idx=30, max_roi_calls=None):
    if not os.path.exists(input_path):
        print(f"Error: input file not found at {input_path}")
        sys.exit(1)

    print(f"Processing input: {input_path}")
    frame = load_frame(input_path, frame_idx=frame_idx)

    extracted_path = OUTPUT_DIR / "extracted_frame.jpg"
    cv2.imwrite(str(extracted_path), frame)
    print(f"Saved extracted frame to {extracted_path}")

    mapping = run_staged_mapping(frame, max_roi_calls=max_roi_calls)

    mapping_path = OUTPUT_DIR / "detailed_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"Saved detailed mapping to {mapping_path}")

    draw_boxes(frame, mapping, OUTPUT_DIR / "annotated_floor_map.jpg")
    print("Process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a staged CCTV spatial map using Gemini.")
    parser.add_argument("input_path", help="Path to a video or image file")
    parser.add_argument("--frame-idx", type=int, default=30, help="Video frame index to extract")
    parser.add_argument(
        "--max-roi-calls",
        type=int,
        default=None,
        help="Limit ROI calls for quick prompt iteration",
    )
    args = parser.parse_args()
    main(args.input_path, frame_idx=args.frame_idx, max_roi_calls=args.max_roi_calls)
