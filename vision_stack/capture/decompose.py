import cv2
import os
import argparse

def decompose(video_path, output_dir, fmt="jpg", every_n=1):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open: {video_path}")
        raise SystemExit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video:  {width}x{height} @ {fps:.1f}fps  |  {total_frames} frames total")
    print(f"Output: {output_dir}/  |  format={fmt}  |  sampling=every {every_n} frame(s)")

    saved = 0
    index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if index % every_n == 0:
            filename = os.path.join(output_dir, f"frame_{index:05d}.{fmt}")
            cv2.imwrite(filename, frame)
            saved += 1

        index += 1

    cap.release()
    print(f"Done. Saved {saved} frames to '{output_dir}/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose an avi into frames.")
    parser.add_argument("video",      help="Path to input .avi file")
    parser.add_argument("-o",         dest="output_dir", default="frames",  help="Output directory (default: frames/)")
    parser.add_argument("-f",         dest="fmt",        default="jpg",     help="Image format: jpg or png (default: jpg)")
    parser.add_argument("-n",         dest="every_n",    default=1, type=int, help="Save every Nth frame (default: 1)")
    args = parser.parse_args()

    decompose(args.video, args.output_dir, args.fmt, args.every_n)