import argparse
import base64
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image


OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_GRID_COLUMNS = 26
DEFAULT_GRID_ROWS = 20
MODEL_PRESETS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def fail_setup(message):
    print(message, file=sys.stderr)
    print(
        "\nUse the isolated SAM environment:\n"
        "  .\\.venv-sam\\Scripts\\python.exe -m pip install -r experiment\\cctv_intelligence\\requirements-sam.txt\n",
        file=sys.stderr,
    )
    sys.exit(2)


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def safe_name(value):
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or "cctv_map"


def load_input_frame(input_path, frame_idx, output_frame_path):
    path = Path(input_path)
    if not path.exists():
        raise ValueError(f"Input not found: {path}")

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read image: {path}")
        cv2.imwrite(str(output_frame_path), frame)
        return frame, "image", None

    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported input type: {path.suffix}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_idx is None:
        frame_idx = max(0, total_frames // 2) if total_frames else 30

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_idx} from video: {path}")

    cv2.imwrite(str(output_frame_path), frame)
    return frame, "video", int(frame_idx)


def parse_json_response(text):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


def load_gemini_client(timeout_seconds):
    load_dotenv(OUTPUT_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Create experiment/cctv_intelligence/.env")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        fail_setup(f"google-genai is not installed: {exc}")
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_seconds * 1000)), types


def load_openai_client(timeout_seconds):
    load_dotenv(OUTPUT_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to experiment/cctv_intelligence/.env")
    try:
        from openai import OpenAI
    except ImportError as exc:
        fail_setup(f"openai is not installed: {exc}")
    return OpenAI(api_key=api_key, timeout=timeout_seconds)


def load_sam2_predictor(model_size, checkpoint, model_cfg, device):
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        fail_setup(f"SAM2 is not installed: {exc}")

    if model_size:
        checkpoint_name, preset_cfg = MODEL_PRESETS[model_size]
        checkpoint = checkpoint or str(OUTPUT_DIR / "models" / checkpoint_name)
        model_cfg = model_cfg or preset_cfg

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        fail_setup(f"SAM2 checkpoint not found: {checkpoint_path}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_sam2(model_cfg, str(checkpoint_path), device=device)
    return SAM2ImagePredictor(model), torch, str(checkpoint_path), model_cfg, device


def image_to_pil(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def resize_for_gemini(frame_bgr, max_size):
    if not max_size:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return frame_bgr
    scale = max_size / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def column_name(index):
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def column_index(name):
    value = 0
    for char in name.upper():
        if not ("A" <= char <= "Z"):
            raise ValueError(f"Invalid grid column: {name}")
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def cell_bounds(cell, frame_shape, columns, rows):
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", str(cell).strip())
    if not match:
        raise ValueError(f"Invalid grid cell: {cell}")
    col = column_index(match.group(1))
    row = int(match.group(2)) - 1
    h, w = frame_shape[:2]
    if col < 0 or col >= columns or row < 0 or row >= rows:
        raise ValueError(f"Grid cell out of bounds: {cell}")
    x1 = int(round((col / columns) * w))
    x2 = int(round(((col + 1) / columns) * w))
    y1 = int(round((row / rows) * h))
    y2 = int(round(((row + 1) / rows) * h))
    return [x1, y1, x2, y2]


def union_boxes(boxes):
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def expand_box(box, frame_shape, padding_ratio):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    return [
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(w, int(round(x2 + pad_x))),
        min(h, int(round(y2 + pad_y))),
    ]


def cells_to_box(cells, frame_shape, columns, rows, padding_ratio):
    boxes = [cell_bounds(cell, frame_shape, columns, rows) for cell in cells]
    return expand_box(union_boxes(boxes), frame_shape, padding_ratio)


def location_to_box(location, frame_shape, columns, rows, padding_ratio):
    top_left = location.get("top_left_cell")
    bottom_right = location.get("bottom_right_cell")
    if top_left and bottom_right:
        x1a, y1a, _, _ = cell_bounds(top_left, frame_shape, columns, rows)
        _, _, x2b, y2b = cell_bounds(bottom_right, frame_shape, columns, rows)
        return expand_box([x1a, y1a, x2b, y2b], frame_shape, padding_ratio)

    cells = location.get("grid_cells") or []
    if not cells:
        raise ValueError("location has no corner cells or grid_cells")
    return cells_to_box(cells, frame_shape, columns, rows, padding_ratio)


def cell_point(cell, frame_shape, columns, rows, offset=None):
    x1, y1, x2, y2 = cell_bounds(cell, frame_shape, columns, rows)
    offset = offset if isinstance(offset, list) and len(offset) == 2 else [50, 50]
    ox = max(0, min(100, float(offset[0]))) / 100.0
    oy = max(0, min(100, float(offset[1]))) / 100.0
    return [x1 + (x2 - x1) * ox, y1 + (y2 - y1) * oy]


def draw_grid(frame_bgr, columns, rows, output_path):
    grid = frame_bgr.copy()
    h, w = grid.shape[:2]
    line_color = (0, 255, 255)
    label_bg = (0, 0, 0)
    label_fg = (255, 255, 255)

    overlay = grid.copy()
    for col in range(columns + 1):
        x = int(round((col / columns) * w))
        cv2.line(overlay, (x, 0), (x, h), line_color, 1)
    for row in range(rows + 1):
        y = int(round((row / rows) * h))
        cv2.line(overlay, (0, y), (w, y), line_color, 1)
    grid = cv2.addWeighted(overlay, 0.55, grid, 0.45, 0)

    for col in range(columns):
        x = int(round((col / columns) * w)) + 4
        cv2.rectangle(grid, (x - 2, 4), (x + 42, 28), label_bg, -1)
        cv2.putText(grid, column_name(col), (x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_fg, 2)

    for row in range(rows):
        y = int(round((row / rows) * h)) + 24
        cv2.rectangle(grid, (4, y - 20), (54, y + 4), label_bg, -1)
        cv2.putText(grid, str(row + 1), (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_fg, 2)

    cv2.imwrite(str(output_path), grid)
    return grid


def generate_json(client, types, model, image_bgr, prompt, stage_name):
    print(f"Gemini stage started: {stage_name}", flush=True)
    response = client.models.generate_content(
        model=model,
        contents=[image_to_pil(image_bgr), prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
    )
    print(f"Gemini stage finished: {stage_name}", flush=True)
    return parse_json_response(response.text)


def image_bgr_to_data_url(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def generate_json_openai(client, model, image_bgr, prompt, stage_name):
    print(f"OpenAI stage started: {stage_name}", flush=True)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_bgr_to_data_url(image_bgr)},
                ],
            }
        ],
    )
    print(f"OpenAI stage finished: {stage_name}", flush=True)
    return parse_json_response(response.output_text)


def discover_objects(client, types, provider, model, frame_bgr, max_objects):
    prompt = f"""
You are creating a reusable high-level CCTV map for an industrial/factory camera.

First understand the scene:
- What kind of factory/work area is visible?
- What fixed things would help a future system answer "where was the person/event?"
- What visible objects are stable enough to be reused as map anchors across the day?

Your job is not to list every object. Your job is to choose the smallest useful set of permanent map anchors.

End goal:
Later, people will be detected in the CCTV feed. Their foot position will be matched against this map to answer high-level questions like:
- "Which area was the person in?"
- "Was the person near a machine, workstation, aisle, or restricted zone?"
- "Where did the person move before/after an event?"

Therefore, include an anchor only if it improves location understanding for people/events.

Include only stable, useful map anchors:
- fixed machines or production equipment
- individual workstations, benches, tables, inspection stations, packing stations
- fixed control panels, electrical panels, mounted HMIs, gauges, button panels
- storage racks, cabinets, cages, fixed material stands
- doors, entry/exit points, gates, fixed barriers
- painted safety lines, marked zones, walkways, aisles, if they define where people move
- fixed signage/nameplates if they identify an area or machine
- broad named zones only when individual objects are too far/small/unclear to map reliably

Exclude transient or low-value objects:
- people, body parts, PPE worn by people
- loose crates, bins, trays, tools, parts, raw material, finished goods, scrap
- loose papers/documents unless they are on a fixed board/holder
- shadows, glare, reflections, floor stains, wall/ceiling/roof/background
- generic support frames when a clearer machine/station/panel anchor exists
- tiny far-background objects that do not help localize a person
- duplicate anchors that heavily overlap a better anchor
- decorative or structural objects unless they are useful location landmarks. If included only as a landmark, say so clearly in the description.

CRITICAL INSTANCE RULES:
- Return unique physical instances, not broad groups.
- Repeated similar objects are important separate anchors. Return each visible instance separately.
- Do not group a row/line/area if individual machines or stations can be distinguished.
- Use visible labels/text/numbers when present to name each instance.
- If labels are not readable, use stable relative names: left, center-left, center, center-right, right, front, rear.
- If a large station has a useful fixed control panel, return the station and the panel as separate anchors only when both help location.
- Each returned item must be localizable later by a compact grid box or compact polygon. If it would need a huge strip, split it or skip it.
- Prefer anchors that divide the camera view into meaningful places where humans/events can occur.
- Prefer nearer/larger/person-relevant anchors over small far-background equipment.
- Avoid returning both a broad zone and the individual anchors inside it, unless the broad zone is a walkway/aisle/safety zone.
- For low-resolution or distant CCTV views, return fewer stronger zones instead of many tiny uncertain objects.

Selection strategy:
1. Include primary movement areas first: aisles, walkways, entry/exit paths, restricted/safety boundaries.
2. Include major fixed machines/workstations that define where people work.
3. Include control/electrical panels only if they are distinct interaction points.
4. Include storage/racks/doors only if they define a meaningful area.
5. Stop before adding weak, redundant, tiny, or overlapping anchors.

Description writing rules:
- The description is stored in the map database and later used by an agent to explain worker movement near this mapped object/area.
- Do not write only what it looks like. Include the likely factory purpose when visible or reasonably inferable.
- If workers likely interact with it, describe the interaction cautiously: "workers may stand nearby to..." or "appears to be used for..."
- If it is only a landmark/reference object, say it is only useful for describing location.
- Never claim that nearby foot positions prove a worker operated the machine, performed a task, belonged to that area, or interacted with the object. Say that exact activity needs visual confirmation.

Return at most {max_objects} items, but use fewer if fewer are actually useful. Order by usefulness for person/event location.

Return JSON:
{{
  "scene_summary": {{
    "place_type": "short generic scene type",
    "map_strategy": "short explanation of what anchors matter in this view",
    "image_quality": "high|medium|low",
    "recommended_anchor_density": "detailed|moderate|sparse"
  }},
  "items": [
    {{
      "id": "short_snake_case_id",
      "label": "human readable name",
      "type": "machine|control_panel|workstation|walkway|safety_boundary|fixed_signage|electrical_panel|storage_area|unknown_anchor",
      "description": "factory-aware description: what this mapped object/area is, what it appears to be normally used for, how worker movement near it should be described, and what must not be concluded from nearby foot positions alone. If it is only a location landmark, say that clearly.",
      "visual_disambiguation": "how to recognize this exact instance versus neighboring similar objects",
      "why_map_anchor": "why this helps locate future people/events",
      "importance": "high|medium|low"
    }}
  ]
}}
"""
    if provider == "openai":
        return generate_json_openai(client, model, frame_bgr, prompt, "object discovery").get("items", [])
    return generate_json(client, types, model, frame_bgr, prompt, "object discovery").get("items", [])


def locate_objects_on_grid(client, types, provider, model, grid_bgr, items, columns, rows):
    object_list = json.dumps(
        [
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "type": item.get("type"),
                "description": item.get("description"),
            }
            for item in items
        ],
        indent=2,
    )
    prompt = f"""
This image has a visible grid with columns {column_name(0)}-{column_name(columns - 1)} and rows 1-{rows}.

For each target object below, identify:
1. the top-left grid cell of a compact high-level map box around the visible stable anchor/zone
2. the bottom-right grid cell of that compact high-level map box
3. the single best center cell for a click/point prompt or map marker
4. the center offset inside that center cell as [x_percent, y_percent], where 50,50 is the cell center

Target objects:
{object_list}

Rules:
- Use only visible grid cell IDs like A1, B7, Z20.
- Return only corner cells, not a long list of every covered cell.
- Use the smallest practical map box that still covers the useful anchor/zone.
- Do not include huge surrounding background, people, loose material, or neighboring anchors.
- For machines, box the visible machine body/nameplate/control body plus immediate operating face, not the entire factory bay.
- For workstations, box the fixed station/bench/frame footprint as a high-level zone where a person would work, but avoid unrelated neighboring stations.
- For control/electrical panels, box only the fixed panel body and its immediate interaction area.
- For walkways/safety boundaries, cover the visible marked path/line/zone segment that best represents the map anchor.
- For repeated similar objects, keep each location separate and compact.
- If an object cannot be located confidently, set grid_cells to [] and center_cell to null.

Return JSON:
{{
  "locations": [
    {{
      "id": "same id from target object",
      "top_left_cell": "A1",
      "bottom_right_cell": "B2",
      "center_cell": "A1",
      "center_offset": [50, 50],
      "reason": "short reason",
      "confidence": 0.0
    }}
  ]
}}
"""
    if provider == "openai":
        return generate_json_openai(client, model, grid_bgr, prompt, "grid location").get("locations", [])
    return generate_json(client, types, model, grid_bgr, prompt, "grid location").get("locations", [])


def contour_to_polygon(mask, epsilon_ratio=0.006):
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return [], None
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    polygon = [[int(point[0][0]), int(point[0][1])] for point in approx]
    x, y, w, h = cv2.boundingRect(contour)
    return polygon, [int(x), int(y), int(x + w), int(y + h)]


def choose_mask(masks, scores):
    if len(masks) == 0:
        return None, None
    best_index = int(np.argmax(scores))
    return masks[best_index], float(scores[best_index])


def quality_flags(sam_score, polygon, location_confidence):
    flags = []
    if sam_score is not None and sam_score < 0.75:
        flags.append("low_sam_score")
    if len(polygon) > 80:
        flags.append("complex_mask_boundary")
    if location_confidence is not None and location_confidence < 0.65:
        flags.append("low_grid_location_confidence")
    return flags


def run_sam(predictor, image_rgb, objects, locations, frame_shape, columns, rows, padding_ratio):
    predictor.set_image(image_rgb)
    locations_by_id = {item.get("id"): item for item in locations}
    results = []

    for obj in objects:
        location = locations_by_id.get(obj.get("id"))
        has_box = bool((location or {}).get("top_left_cell") and (location or {}).get("bottom_right_cell")) or bool(
            (location or {}).get("grid_cells")
        )
        if not location or not has_box or not location.get("center_cell"):
            results.append({**obj, "needs_review": True, "quality_flags": ["not_located_on_grid"], "source": {"grid_location": location}})
            continue

        try:
            prompt_box = location_to_box(location, frame_shape, columns, rows, padding_ratio)
            prompt_point = cell_point(location["center_cell"], frame_shape, columns, rows, location.get("center_offset"))
        except ValueError as exc:
            results.append({**obj, "needs_review": True, "quality_flags": [str(exc)], "source": {"grid_location": location}})
            continue

        masks, scores, _ = predictor.predict(
            point_coords=np.array([prompt_point], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            box=np.array(prompt_box, dtype=np.float32),
            multimask_output=True,
        )
        mask, sam_score = choose_mask(masks, scores)
        if mask is None:
            results.append({**obj, "needs_review": True, "quality_flags": ["sam_returned_no_mask"], "source": {"grid_location": location}})
            continue

        polygon, tight_box = contour_to_polygon(mask)
        if not polygon or tight_box is None:
            results.append({**obj, "needs_review": True, "quality_flags": ["empty_mask_polygon"], "source": {"grid_location": location}})
            continue

        confidence = location.get("confidence")
        flags = quality_flags(sam_score, polygon, confidence)
        results.append(
            {
                **obj,
                "geometry_type": "polygon",
                "points": polygon,
                "box": tight_box,
                "needs_review": bool(flags),
                "quality_flags": flags,
                "source": {
                    "semantic_source": "gemini_object_discovery",
                    "location_source": "gemini_grid_prompt",
                    "geometry_source": "sam2_point_box",
                    "grid_location": location,
                    "prompt_point": [round(float(prompt_point[0]), 2), round(float(prompt_point[1]), 2)],
                    "prompt_box": prompt_box,
                    "sam_score": sam_score,
                },
            }
        )

    return results


def run_grid_boxes_only(objects, locations, frame_shape, columns, rows, padding_ratio):
    locations_by_id = {item.get("id"): item for item in locations}
    results = []

    for obj in objects:
        location = locations_by_id.get(obj.get("id"))
        has_box = bool((location or {}).get("top_left_cell") and (location or {}).get("bottom_right_cell")) or bool(
            (location or {}).get("grid_cells")
        )
        if not location or not has_box or not location.get("center_cell"):
            results.append(
                {
                    **obj,
                    "geometry_type": "box",
                    "needs_review": True,
                    "quality_flags": ["not_located_on_grid"],
                    "source": {"grid_location": location},
                }
            )
            continue

        try:
            box = location_to_box(location, frame_shape, columns, rows, padding_ratio)
            point = cell_point(location["center_cell"], frame_shape, columns, rows, location.get("center_offset"))
        except ValueError as exc:
            results.append(
                {
                    **obj,
                    "geometry_type": "box",
                    "needs_review": True,
                    "quality_flags": [str(exc)],
                    "source": {"grid_location": location},
                }
            )
            continue

        x1, y1, x2, y2 = box
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        confidence = location.get("confidence")
        flags = []
        if confidence is not None and confidence < 0.65:
            flags.append("low_grid_location_confidence")

        results.append(
            {
                **obj,
                "geometry_type": "box",
                "points": points,
                "box": box,
                "center_point": [round(float(point[0]), 2), round(float(point[1]), 2)],
                "needs_review": bool(flags),
                "quality_flags": flags,
                "source": {
                    "semantic_source": "vision_object_discovery",
                    "location_source": "vision_grid_prompt",
                    "geometry_source": "grid_corner_box",
                    "grid_location": location,
                    "prompt_point": [round(float(point[0]), 2), round(float(point[1]), 2)],
                    "prompt_box": box,
                },
            }
        )

    return results


def draw_preview(frame_bgr, map_data, output_path):
    preview = frame_bgr.copy()
    legend = []
    for index, item in enumerate(map_data["items"], start=1):
        if item.get("points"):
            points = np.array(item["points"], dtype=np.int32)
            color = (0, 0, 255) if item.get("needs_review") else (0, 255, 255)
            cv2.polylines(preview, [points], isClosed=True, color=color, thickness=3)
            if item.get("center_point"):
                cx, cy = item["center_point"]
                cv2.circle(preview, (int(cx), int(cy)), 7, (0, 0, 255), -1)
            x1, y1, _, _ = item.get("box", [0, 0, 0, 0])
        else:
            color = (0, 0, 255)
            x1, y1 = 10, 30 + index * 24

        cv2.rectangle(preview, (x1, max(0, y1 - 22)), (x1 + 34, y1), color, -1)
        cv2.putText(preview, str(index), (x1 + 6, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        review = " REVIEW" if item.get("needs_review") else ""
        legend.append(f"{index}. {item.get('label')} [{item.get('type')}]{review}")

    cv2.imwrite(str(output_path), preview)
    Path(output_path).with_suffix(".legend.txt").write_text("\n".join(legend), encoding="utf-8")


def draw_location_preview(frame_bgr, objects, locations, frame_shape, columns, rows, padding_ratio, output_path):
    preview = frame_bgr.copy()
    objects_by_id = {item.get("id"): item for item in objects}
    legend = []

    for index, location in enumerate(locations, start=1):
        obj = objects_by_id.get(location.get("id"), {})
        has_box = bool(location.get("top_left_cell") and location.get("bottom_right_cell")) or bool(location.get("grid_cells"))
        if not has_box:
            legend.append(f"{index}. {obj.get('label', location.get('id'))} NOT LOCATED")
            continue

        try:
            box = location_to_box(location, frame_shape, columns, rows, padding_ratio)
            point = cell_point(location.get("center_cell"), frame_shape, columns, rows, location.get("center_offset"))
        except Exception as exc:
            legend.append(f"{index}. {obj.get('label', location.get('id'))} ERROR {exc}")
            continue

        x1, y1, x2, y2 = box
        color = (255, 0, 255)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
        cv2.circle(preview, (int(point[0]), int(point[1])), 8, (0, 0, 255), -1)
        cv2.rectangle(preview, (x1, max(0, y1 - 22)), (x1 + 34, y1), color, -1)
        cv2.putText(preview, str(index), (x1 + 6, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        if location.get("top_left_cell") and location.get("bottom_right_cell"):
            box_text = f"{location.get('top_left_cell')}->{location.get('bottom_right_cell')}"
        else:
            box_text = ",".join(location.get("grid_cells") or [])
        legend.append(f"{index}. {obj.get('label', location.get('id'))} box={box_text} center={location.get('center_cell')}")

    cv2.imwrite(str(output_path), preview)
    Path(output_path).with_suffix(".legend.txt").write_text("\n".join(legend), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build a CCTV map using Gemini grid prompting plus SAM2 point/box masks.")
    parser.add_argument("run_name", nargs="?", default=None, help="Short name for this mapping run, e.g. factory_cam_01")
    parser.add_argument("input_path", nargs="?", default=None, help="Image or video path")
    parser.add_argument("--image", default=None, help="Backward-compatible image input")
    parser.add_argument("--frame-idx", type=int, default=None, help="Video frame index. Defaults to middle frame when possible.")
    parser.add_argument("--runs-dir", default=str(OUTPUT_DIR / "runs"))
    parser.add_argument("--camera-id", default="cam_moulding_01")
    parser.add_argument("--vision-provider", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))
    parser.add_argument("--gemini-timeout", type=int, default=45)
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_VISION_MODEL", "gpt-5.5"))
    parser.add_argument("--openai-timeout", type=int, default=90)
    parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="large")
    parser.add_argument("--use-sam", action="store_true", help="Run SAM2 mask refinement. Default is fast grid-box mapping.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-cfg", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--columns", type=int, default=DEFAULT_GRID_COLUMNS)
    parser.add_argument("--rows", type=int, default=DEFAULT_GRID_ROWS)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--gemini-max-size", type=int, default=1200)
    parser.add_argument("--box-padding", type=float, default=0.08)
    parser.add_argument("--objects-json", default=None, help="Optional existing discovery JSON to skip discovery call")
    parser.add_argument("--locations-json", default=None, help="Optional existing location JSON to skip grid location call")
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--grid-output", default=str(OUTPUT_DIR / "grid_frame.jpg"))
    parser.add_argument("--discovery-output", default=str(OUTPUT_DIR / "grid_object_discovery.json"))
    parser.add_argument("--locations-output", default=str(OUTPUT_DIR / "grid_locations.json"))
    parser.add_argument("--location-preview", default=str(OUTPUT_DIR / "grid_location_preview.jpg"))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "grid_camera_map.json"))
    parser.add_argument("--preview", default=str(OUTPUT_DIR / "grid_camera_map_preview.jpg"))
    parser.add_argument("--grid-only", action="store_true")
    args = parser.parse_args()

    # Smart argument shift: If user provided 1 argument and it's an existing file, treat it as input_path
    if args.run_name and not args.input_path and Path(args.run_name).exists():
        args.input_path = args.run_name
        args.run_name = None

    input_path = args.input_path or args.image or str(OUTPUT_DIR / "extracted_frame.jpg")
    run_name = safe_name(args.run_name or Path(input_path).stem)
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.camera_id == "cam_moulding_01" and args.run_name:
        args.camera_id = run_name

    if args.grid_output == str(OUTPUT_DIR / "grid_frame.jpg"):
        args.grid_output = str(run_dir / "grid_frame.jpg")
    if args.discovery_output == str(OUTPUT_DIR / "grid_object_discovery.json"):
        args.discovery_output = str(run_dir / "grid_object_discovery.json")
    if args.locations_output == str(OUTPUT_DIR / "grid_locations.json"):
        args.locations_output = str(run_dir / "grid_locations.json")
    if args.location_preview == str(OUTPUT_DIR / "grid_location_preview.jpg"):
        args.location_preview = str(run_dir / "grid_location_preview.jpg")
    if args.output == str(OUTPUT_DIR / "grid_camera_map.json"):
        args.output = str(run_dir / "grid_camera_map.json")
    if args.preview == str(OUTPUT_DIR / "grid_camera_map_preview.jpg"):
        args.preview = str(run_dir / "grid_camera_map_preview.jpg")

    extracted_frame_path = run_dir / "extracted_frame.jpg"
    frame_bgr, input_type, used_frame_idx = load_input_frame(input_path, args.frame_idx, extracted_frame_path)
    print(f"Loaded {input_type}: {input_path}", flush=True)
    if used_frame_idx is not None:
        print(f"Extracted frame index: {used_frame_idx}", flush=True)
    print(f"Saved extracted frame: {extracted_frame_path}", flush=True)

    grid_bgr = draw_grid(frame_bgr, args.columns, args.rows, args.grid_output)
    if args.grid_only:
        print(f"Saved grid image: {args.grid_output}")
        return

    if args.vision_provider == "openai":
        client = load_openai_client(args.openai_timeout)
        types = None
        vision_model = args.openai_model
    else:
        client, types = load_gemini_client(args.gemini_timeout)
        vision_model = args.gemini_model

    gemini_frame_bgr = resize_for_gemini(frame_bgr, args.gemini_max_size)
    gemini_grid_bgr = resize_for_gemini(grid_bgr, args.gemini_max_size)
    print(
        f"Vision provider: {args.vision_provider} model={vision_model}",
        flush=True,
    )
    print(
        f"Vision image size: frame={gemini_frame_bgr.shape[1]}x{gemini_frame_bgr.shape[0]}, "
        f"grid={gemini_grid_bgr.shape[1]}x{gemini_grid_bgr.shape[0]}",
        flush=True,
    )
    objects = read_json(args.objects_json).get("items", []) if args.objects_json else discover_objects(
        client, types, args.vision_provider, vision_model, gemini_frame_bgr, args.max_objects
    )
    write_json(args.discovery_output, {"items": objects})
    print(f"Saved object discovery: {args.discovery_output}", flush=True)
    if args.discovery_only:
        print("Stopped after object discovery", flush=True)
        return

    locations = read_json(args.locations_json).get("locations", []) if args.locations_json else locate_objects_on_grid(
        client, types, args.vision_provider, vision_model, gemini_grid_bgr, objects, args.columns, args.rows
    )
    write_json(args.locations_output, {"locations": locations})
    print(f"Saved grid locations: {args.locations_output}", flush=True)
    draw_location_preview(
        frame_bgr,
        objects,
        locations,
        frame_bgr.shape,
        args.columns,
        args.rows,
        args.box_padding,
        args.location_preview,
    )

    checkpoint = None
    model_cfg = None
    device = None
    if args.use_sam:
        print(f"Loading SAM2 {args.model_size}...", flush=True)
        predictor, torch, checkpoint, model_cfg, device = load_sam2_predictor(
            args.model_size, args.checkpoint, args.model_cfg, args.device
        )
        print(f"SAM2 loaded on {device}. Running masks...", flush=True)
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            items = run_sam(predictor, image_rgb, objects, locations, frame_bgr.shape, args.columns, args.rows, args.box_padding)
        print("SAM2 masks finished", flush=True)
    else:
        print("Skipping SAM2. Using grid corner boxes as final high-level map.", flush=True)
        items = run_grid_boxes_only(objects, locations, frame_bgr.shape, args.columns, args.rows, args.box_padding)

    camera_map = {
        "camera_id": args.camera_id,
        "frame_width": frame_bgr.shape[1],
        "frame_height": frame_bgr.shape[0],
        "map_version": 1,
        "grid": {"columns": args.columns, "rows": args.rows, "image": str(args.grid_output)},
        "items": items,
        "metadata": {
            "pipeline": "vision_grid_prompt_high_level_map",
            "input_path": str(input_path),
            "input_type": input_type,
            "frame_idx": used_frame_idx,
            "extracted_frame": str(extracted_frame_path),
            "vision_provider": args.vision_provider,
            "vision_model": vision_model,
            "use_sam": args.use_sam,
            "sam_model_size": args.model_size if args.use_sam else None,
            "sam_checkpoint": checkpoint,
            "sam_model_cfg": model_cfg,
            "device": device,
        },
    }
    write_json(args.output, camera_map)
    draw_preview(frame_bgr, camera_map, args.preview)
    print(f"Saved grid image: {args.grid_output}")
    print(f"Saved object discovery: {args.discovery_output}")
    print(f"Saved grid locations: {args.locations_output}")
    print(f"Saved grid location preview: {args.location_preview}")
    print(f"Saved camera map: {args.output}")
    print(f"Saved preview: {args.preview}")


if __name__ == "__main__":
    main()
