import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv


DEFAULT_TYPES = {
    "machine",
    "workstation_table",
    "control_panel",
    "storage_rack",
    "safety_line",
    "readable_text_region",
    "fixed_signage",
}

MODEL_PRESETS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


def fail_setup(message):
    print(message, file=sys.stderr)
    print(
        "\nInstall the isolated SAM environment first:\n"
        "  py -3.12 -m venv .venv-sam\n"
        "  .\\.venv-sam\\Scripts\\python.exe -m pip install --upgrade pip\n"
        "  .\\.venv-sam\\Scripts\\python.exe -m pip install -r experiment\\cctv_intelligence\\requirements-sam.txt\n",
        file=sys.stderr,
    )
    sys.exit(2)


def load_sam2_predictor(model_cfg, checkpoint, device):
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        fail_setup(f"SAM2 is not installed in this Python environment: {exc}")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        fail_setup(f"SAM2 checkpoint not found: {checkpoint_path}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sam_model = build_sam2(model_cfg, str(checkpoint_path), device=device)
    predictor = SAM2ImagePredictor(sam_model)
    return predictor, torch


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def scaled_box_to_xyxy(box, frame_shape):
    h, w = frame_shape[:2]
    ymin, xmin, ymax, xmax = box
    return np.array(
        [
            int((xmin / 1000.0) * w),
            int((ymin / 1000.0) * h),
            int((xmax / 1000.0) * w),
            int((ymax / 1000.0) * h),
        ],
        dtype=np.float32,
    )


def xyxy_to_scaled_box(box, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    return [
        int(round((y1 / h) * 1000)),
        int(round((x1 / w) * 1000)),
        int(round((y2 / h) * 1000)),
        int(round((x2 / w) * 1000)),
    ]


def clamp_xyxy(box, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return [x1, y1, x2, y2]


def expand_xyxy(box, frame_shape, padding_ratio):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    return clamp_xyxy([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], frame_shape)


def xyxy_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def parse_json_response(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    return json.loads(stripped)


def load_gemini_client():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to experiment/cctv_intelligence/.env")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        fail_setup(f"google-genai is not installed in .venv-sam: {exc}")
    return genai.Client(api_key=api_key), types


def tighten_box_with_gemini(client, types, crop_rgb, item, gemini_model):
    label = str(item.get("name") or item.get("label") or "target object")
    category = str(item.get("category") or item.get("type") or "object")
    description = str(item.get("description") or "")
    prompt = f"""
You are refining a CCTV map object box inside a cropped image.

Target object:
- label: {label}
- category: {category}
- description: {description}

Return only the tight visible box for this exact target object inside this crop.
If the target is not clearly visible or the crop contains multiple confusing connected structures, return null.

Rules:
- Do not include people, floor, wall, glare, shadows, loose bins, loose trays, or unrelated neighboring machines.
- For machines, prefer the visible machine body/nameplate/control body, not the whole surrounding blue support frame.
- For workstation/control panels, return only the stable panel/body, not papers or loose material.
- Use [ymin, xmin, ymax, xmax] scaled 0-1000 relative to this crop.

Return JSON:
{{"box_2d": [0, 0, 0, 0] | null, "reason": "short reason", "confidence": 0.0}}
"""
    response = client.models.generate_content(
        model=gemini_model,
        contents=[Image.fromarray(crop_rgb), prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
    )
    return parse_json_response(response.text)


def translate_crop_scaled_box_to_frame_xyxy(crop_box_2d, crop_bounds, frame_shape):
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    ymin, xmin, ymax, xmax = crop_box_2d
    x1 = crop_x1 + (xmin / 1000.0) * crop_w
    y1 = crop_y1 + (ymin / 1000.0) * crop_h
    x2 = crop_x1 + (xmax / 1000.0) * crop_w
    y2 = crop_y1 + (ymax / 1000.0) * crop_h
    return clamp_xyxy([x1, y1, x2, y2], frame_shape)


def maybe_tighten_item_box(client, types, image_rgb, frame_shape, item, args):
    original_box = clamp_xyxy(scaled_box_to_xyxy(item["box_2d"], frame_shape), frame_shape)
    crop_bounds = expand_xyxy(original_box, frame_shape, args.gemini_crop_padding)
    cx1, cy1, cx2, cy2 = crop_bounds
    crop_rgb = image_rgb[cy1:cy2, cx1:cx2]

    result = tighten_box_with_gemini(client, types, crop_rgb, item, args.gemini_model)
    crop_box = result.get("box_2d")
    if not isinstance(crop_box, list) or len(crop_box) != 4:
        return original_box, {"used": False, "reason": result.get("reason", "Gemini returned null box")}

    tightened_box = translate_crop_scaled_box_to_frame_xyxy(crop_box, crop_bounds, frame_shape)
    if xyxy_area(tightened_box) < 64:
        return original_box, {"used": False, "reason": "tightened box too small", "raw": result}

    return tightened_box, {"used": True, "raw": result, "crop_bounds": crop_bounds}


def contour_to_polygon(mask, epsilon_ratio=0.006):
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area <= 0:
        return [], None

    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    polygon = [[int(point[0][0]), int(point[0][1])] for point in approx]
    x, y, w, h = cv2.boundingRect(contour)
    return polygon, [int(x), int(y), int(x + w), int(y + h)]


def choose_mask(masks, scores, prompt_box=None):
    if len(masks) == 0:
        return None, None
    if prompt_box is None:
        best_index = int(np.argmax(scores))
    else:
        prompt_area = max(1, xyxy_area(prompt_box))
        ranked = []
        for index, mask in enumerate(masks):
            _, mask_box = contour_to_polygon(mask, epsilon_ratio=0.01)
            mask_area = xyxy_area(mask_box) if mask_box else 0
            area_ratio = mask_area / prompt_area
            area_penalty = abs(np.log(max(area_ratio, 0.05)))
            ranked.append((float(scores[index]) - 0.12 * area_penalty, index))
        best_index = max(ranked)[1]
    return masks[best_index], float(scores[best_index])


def normalize_id(label):
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)
    clean = "_".join(part for part in clean.split("_") if part)
    return clean[:64] or "map_item"


def load_gemini_items(mapping_data, allowed_types):
    items = mapping_data.get("items")
    if items is None:
        items = mapping_data.get("anchors", [])

    result = []
    for item in items:
        category = str(item.get("category") or item.get("type") or "").strip()
        if allowed_types and category not in allowed_types:
            continue
        box = item.get("box_2d")
        if not isinstance(box, list) or len(box) != 4:
            continue
        result.append(item)
    return result


def quality_flags(prompt_box, tight_box, polygon, sam_score):
    flags = []
    prompt_area = max(1, xyxy_area(prompt_box))
    mask_area = max(1, xyxy_area(tight_box))
    ratio = mask_area / prompt_area
    if ratio > 1.35:
        flags.append("mask_much_larger_than_prompt")
    if ratio < 0.12:
        flags.append("mask_much_smaller_than_prompt")
    if len(polygon) > 80:
        flags.append("complex_mask_boundary")
    if sam_score is not None and sam_score < 0.75:
        flags.append("low_sam_score")
    return flags


def refine_items_with_sam(predictor, image_rgb, frame_shape, gemini_items, args):
    predictor.set_image(image_rgb)
    refined = []
    gemini_client = None
    gemini_types = None

    if args.tighten_with_gemini:
        gemini_client, gemini_types = load_gemini_client()

    for index, item in enumerate(gemini_items, start=1):
        label = str(item.get("name") or item.get("label") or f"item_{index}").strip()
        category = str(item.get("category") or item.get("type") or "unknown_anchor").strip()
        original_box = clamp_xyxy(scaled_box_to_xyxy(item["box_2d"], frame_shape), frame_shape)
        input_box = original_box
        tighten_metadata = {"used": False}

        if args.tighten_with_gemini:
            try:
                input_box, tighten_metadata = maybe_tighten_item_box(
                    gemini_client,
                    gemini_types,
                    image_rgb,
                    frame_shape,
                    item,
                    args,
                )
            except Exception as exc:
                tighten_metadata = {"used": False, "reason": f"Gemini tightening failed: {exc}"}

        center_x = (input_box[0] + input_box[2]) / 2.0
        center_y = (input_box[1] + input_box[3]) / 2.0
        point_coords = np.array([[center_x, center_y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords if args.use_center_point else None,
            point_labels=point_labels if args.use_center_point else None,
            box=input_box,
            multimask_output=True,
        )
        mask, sam_score = choose_mask(masks, scores, prompt_box=input_box)
        if mask is None:
            continue

        polygon, tight_box = contour_to_polygon(mask)
        if not polygon or tight_box is None:
            continue
        flags = quality_flags(input_box, tight_box, polygon, sam_score)

        refined.append(
            {
                "id": normalize_id(label),
                "label": label,
                "type": category,
                "geometry_type": "polygon",
                "points": polygon,
                "box": tight_box,
                "needs_review": bool(flags),
                "quality_flags": flags,
                "source": {
                    "semantic_source": "gemini",
                    "geometry_source": "sam2",
                    "gemini_box_2d": item["box_2d"],
                    "prompt_box": input_box,
                    "prompt_box_2d": xyxy_to_scaled_box(input_box, frame_shape),
                    "gemini_tightening": tighten_metadata,
                    "gemini_confidence": item.get("confidence"),
                    "sam_score": sam_score,
                },
                "importance": "high" if category in {"machine", "workstation_table"} else "medium",
                "description": item.get("description", ""),
            }
        )

    return refined


def draw_preview(frame_bgr, map_data, output_path):
    preview = frame_bgr.copy()
    for index, item in enumerate(map_data["items"], start=1):
        points = np.array(item["points"], dtype=np.int32)
        color = (0, 255, 255)
        cv2.polylines(preview, [points], isClosed=True, color=color, thickness=3)

        x1, y1, _, _ = item["box"]
        badge = str(index)
        cv2.rectangle(preview, (x1, max(0, y1 - 22)), (x1 + 34, y1), color, -1)
        cv2.putText(preview, badge, (x1 + 6, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    legend_path = Path(output_path).with_suffix(".legend.txt")
    legend_path.write_text(
        "\n".join(f"{idx}. {item['label']} [{item['type']}]" for idx, item in enumerate(map_data["items"], start=1)),
        encoding="utf-8",
    )
    cv2.imwrite(str(output_path), preview)


def build_camera_map(args):
    image_path = Path(args.image)
    mapping_path = Path(args.mapping)
    output_path = Path(args.output)
    preview_path = Path(args.preview)

    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(image_rgb)  # Validate Pillow can handle the image in this environment.

    mapping_data = read_json(mapping_path)
    allowed_types = set(args.types.split(",")) if args.types else DEFAULT_TYPES
    gemini_items = load_gemini_items(mapping_data, allowed_types)
    if not gemini_items:
        raise ValueError(f"No usable Gemini items found in {mapping_path}")

    if args.model_size:
        checkpoint_name, model_cfg = MODEL_PRESETS[args.model_size]
        if not args.checkpoint:
            args.checkpoint = str(Path(__file__).resolve().parent / "models" / checkpoint_name)
        if not args.model_cfg:
            args.model_cfg = model_cfg

    predictor, torch = load_sam2_predictor(args.model_cfg, args.checkpoint, args.device)

    with torch.inference_mode():
        refined_items = refine_items_with_sam(predictor, image_rgb, frame_bgr.shape, gemini_items, args)

    h, w = frame_bgr.shape[:2]
    camera_map = {
        "camera_id": args.camera_id,
        "frame_width": w,
        "frame_height": h,
        "map_version": 1,
        "items": refined_items,
        "metadata": {
            "semantic_source": str(mapping_path),
            "geometry_model": "sam2",
            "model_cfg": args.model_cfg,
            "checkpoint": str(args.checkpoint),
            "tighten_with_gemini": args.tighten_with_gemini,
            "gemini_model": args.gemini_model if args.tighten_with_gemini else None,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, camera_map)
    draw_preview(frame_bgr, camera_map, preview_path)

    print(f"Saved camera map: {output_path}")
    print(f"Saved preview: {preview_path}")
    print(f"Refined {len(refined_items)} of {len(gemini_items)} Gemini items")


def main():
    parser = argparse.ArgumentParser(description="Refine Gemini CCTV map detections using SAM2 masks.")
    parser.add_argument("--image", default="experiment/cctv_intelligence/extracted_frame.jpg")
    parser.add_argument("--mapping", default="experiment/cctv_intelligence/detailed_mapping.json")
    parser.add_argument("--camera-id", default="cam_moulding_01")
    parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default=None)
    parser.add_argument("--checkpoint", default=None, help="Path to SAM2 checkpoint .pt file")
    parser.add_argument("--model-cfg", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--types", default="machine,workstation_table,control_panel,storage_rack,fixed_signage")
    parser.add_argument("--use-center-point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tighten-with-gemini", action="store_true")
    parser.add_argument("--gemini-model", default="gemini-3-flash-preview")
    parser.add_argument("--gemini-crop-padding", type=float, default=0.15)
    parser.add_argument("--output", default="experiment/cctv_intelligence/camera_map.json")
    parser.add_argument("--preview", default="experiment/cctv_intelligence/camera_map_preview.jpg")
    args = parser.parse_args()
    if not args.model_size and (not args.checkpoint or not args.model_cfg):
        parser.error("Provide either --model-size or both --checkpoint and --model-cfg")
    build_camera_map(args)


if __name__ == "__main__":
    main()
