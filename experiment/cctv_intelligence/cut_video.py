import argparse
import subprocess
import os
from pathlib import Path

def format_time_for_filename(time_str):
    """Converts '00:01:30' or '01:30' to '00-01-30' for safe filenames."""
    return time_str.replace(":", "-")

def main():
    parser = argparse.ArgumentParser(description="Cut a segment of video using FFmpeg.")
    parser.add_argument("video_path", help="Path to the source video file")
    parser.add_argument("start_time", help="Start time (e.g., '00:00' or '00:01:30')")
    parser.add_argument("end_time", help="End time (e.g., '01:00' or '00:02:45')")
    parser.add_argument("--out-dir", default="cut_videos", help="Output directory name (created if doesn't exist)")
    
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file '{video_path}' does not exist.")
        return

    # Create output directory relative to where the script is run (or inside cctv_intelligence)
    out_dir = Path(__file__).resolve().parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename: original_name_cut_00-00_to_01-00.mp4
    safe_start = format_time_for_filename(args.start_time)
    safe_end = format_time_for_filename(args.end_time)
    out_filename = f"{video_path.stem}_cut_{safe_start}_to_{safe_end}{video_path.suffix}"
    out_filepath = out_dir / out_filename

    print(f"Cutting video from {args.start_time} to {args.end_time}...")
    print(f"Source: {video_path}")
    print(f"Output: {out_filepath}")

    # FFmpeg command
    # -i input
    # -ss start_time
    # -to end_time
    # -c copy (copies streams without re-encoding for blazing fast speed, cuts on keyframes)
    cmd = [
        "ffmpeg",
        "-y", # Overwrite output file if exists
        "-i", str(video_path),
        "-ss", args.start_time,
        "-to", args.end_time,
        "-c", "copy",
        "-an", # Drop audio to prevent codec compatibility issues
        str(out_filepath)
    ]

    try:
        # Run FFmpeg
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"\n[SUCCESS] Video successfully cut and saved to:\n{out_filepath}")
    except subprocess.CalledProcessError as e:
        print("\n[ERROR] FFmpeg failed!")
        print("Error details:")
        print(e.stderr)

if __name__ == "__main__":
    main()
