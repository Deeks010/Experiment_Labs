import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def draw_boxes(frame, boxes, scores, labels, threshold):
    """Draw bounding boxes on a frame using absolute coordinates."""
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x0, y0, x1, y1 = [int(v) for v in box]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(frame, text, (x0, max(0, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def main():
    parser = argparse.ArgumentParser(description="Run Grounding DINO on a video and output an annotated video.")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("prompt", type=str, help="Text prompt describing objects to detect (e.g., 'bag sack')")
    parser.add_argument("--output", type=str, default=None, help="Path to output annotated video (default: <video>_annotated.mp4)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (cuda or cpu). Auto-detect if omitted.")
    parser.add_argument("--save_frames", action="store_true", help="If set, saves each annotated frame as an image in a 'frames' subfolder.")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path = Path(args.output) if args.output else video_path.with_name(video_path.stem + "_annotated.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # FIX 1: Use tiny model to match the test script so it doesn't re-download and runs faster
    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"Loading Grounding DINO model ({model_id})…")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)

    # Format the prompt correctly (lowercase, ending with .)
    text_prompt = args.prompt.lower()
    if not text_prompt.endswith("."):
        text_prompt += " ."

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    frames_dir = None
    if args.save_frames:
        frames_dir = video_path.parent / f"{video_path.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing frames")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # FIX 2: Use the robust post-processing API that works with all transformers versions
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                target_sizes=[pil_image.size[::-1]]
            )[0]
            
            boxes = results["boxes"].cpu().numpy()
            labels = results["labels"]
            scores = results["scores"].cpu().numpy()
            
            # Manually filter by threshold
            filtered_indices = [i for i, score in enumerate(scores) if score >= args.threshold]
            filtered_boxes = [boxes[i] for i in filtered_indices]
            filtered_labels = [labels[i] for i in filtered_indices]
            filtered_scores = [scores[i] for i in filtered_indices]
            
            annotated = draw_boxes(frame, filtered_boxes, filtered_scores, filtered_labels, args.threshold)
            out.write(annotated)
            
            if args.save_frames:
                frame_file = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_file), annotated)
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        out.release()
        print(f"\nProcessing complete. Annotated video saved to {output_path}")
        if args.save_frames:
            print(f"Frames saved to {frames_dir}")

if __name__ == "__main__":
    main()
