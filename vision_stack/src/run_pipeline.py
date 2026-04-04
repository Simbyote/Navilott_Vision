"""
run_pipeline.py
===============
Runs the full Phase 2 pipeline on FakeCapture output.
Drop-in for testing on the Pi without a camera.

Usage:
    python run_pipeline.py
    python run_pipeline.py --fps 15 --max-frames 300
"""

import argparse
import time
import sys
import json


from fake_capture import FakeCapture
from linker import load_pipeline_config, run_phase2_on_frame

SAMPLE_DIRS = [
    "vision_stack/sample_img/duckietown/s1",
    "vision_stack/sample_img/duckietown/s2",
    "vision_stack/sample_img/duckietown/s3",
    "vision_stack/sample_img/duckietown/s4",
    "vision_stack/sample_img/duckietown/s5",
]

STATUS_FILE = "/tmp/autobot_status.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps",        type=float, default=30.0)
    parser.add_argument("--max-frames", type=int,   default=None)
    parser.add_argument("--debug-dir",  type=str,   default=None,
                        help="Write debug images here (slow — omit for timing runs)")
    args = parser.parse_args()

    try:
        cfg = load_pipeline_config()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    src = FakeCapture(
        image_dirs = SAMPLE_DIRS,
        target_fps = args.fps,
        loop       = True,
        max_frames = args.max_frames,
    )

    t_start    = time.time()
    count      = 0
    frame_times = []

    try:
        for frame_id, timestamp_ms, frame in src:
            t0 = time.time()

            output = run_phase2_on_frame(
                frame        = frame,
                frame_id     = frame_id,
                timestamp_ms = timestamp_ms,
                cfg          = cfg,
                debug_dir    = args.debug_dir,
                stem         = f"frame_{frame_id:05d}",
            )
            t1       = time.time()
            ms       = (t1 - t0) * 1000
            frame_times.append(ms)

            if frame_id % 30 == 0:
                elapsed  = t1 - t_start
                avg_fps  = round(count / elapsed, 1) if count > 0 else 0.0
                avg_ms   = sum(frame_times[-30:]) / len(frame_times[-30:])
                n_det    = output.detection_count
                print(f"[frame {frame_id:05d}]  "
                      f"pipeline={avg_ms:.1f}ms avg  "
                      f"fps={avg_fps:.1f}  "
                      f"detections={n_det}")
                
            if frame_id % 10 == 0:
                elapsed  = t1 - t_start
                avg_fps  = round(count / elapsed, 1) if count > 0 else 0.0
                avg_ms   = sum(frame_times[-10:]) / len(frame_times[-10:]) if frame_times else 0.0
                status   = {
                    "frame_id":    frame_id,
                    "pipeline_ms": round(ms, 1),
                    "avg_fps":     avg_fps,
                    "detections":  [d.type for d in output.detections],
                    "timestamp":   int(time.time()),
                }
                with open(STATUS_FILE, "w") as f:
                    json.dump(status, f)

            count += 1

    except KeyboardInterrupt:
        print("\nStopped.")

    total = time.time() - t_start
    if frame_times:
        print(f"\nTotal: {count} frames in {total:.1f}s  "
              f"avg={sum(frame_times)/len(frame_times):.1f}ms  "
              f"max={max(frame_times):.1f}ms  "
              f"min={min(frame_times):.1f}ms")



if __name__ == "__main__":
    main()