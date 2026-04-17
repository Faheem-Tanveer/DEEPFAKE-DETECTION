import os
import subprocess
import cv2
from glob import glob


def extract_frames(video_path, out_dir, fps=2):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, int(video_fps / fps))

    frame_id = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            output_path = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(output_path, frame)
            saved += 1

        frame_id += 1

    cap.release()
    return saved


def extract_audio(video_path, out_audio_path, sr=16000):
    os.makedirs(os.path.dirname(out_audio_path), exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        out_audio_path,
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_audio_path


def subject_split_and_process(root_videos, output_data, train_frac=0.8, fps=2, sr=16000):
    """Split subject folder with subdirs 'real' and 'fake' into train/val by subject."""
    os.makedirs(output_data, exist_ok=True)
    for split in ["train", "val"]:
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(output_data, split, "frames", cls), exist_ok=True)
            os.makedirs(os.path.join(output_data, split, "audio", cls), exist_ok=True)

    def process_item(video_path, split, cls):
        filename = os.path.basename(video_path)
        base, _ = os.path.splitext(filename)
        frame_folder = os.path.join(output_data, split, "frames", cls, base)
        audio_file = os.path.join(output_data, split, "audio", cls, f"{base}.wav")
        extract_frames(video_path, frame_folder, fps=fps)
        extract_audio(video_path, audio_file, sr=sr)

    for cls in ["real", "fake"]:
        videos = sorted(glob(os.path.join(root_videos, cls, "*.mp4")))
        subjects = {}
        for v in videos:
            # heuristic: subject ID in filename before underscore (e.g., subject01_video1.mp4)
            sub = os.path.basename(v).split("_")[0]
            subjects.setdefault(sub, []).append(v)

        subject_keys = sorted(subjects.keys())
        split_point = int(len(subject_keys) * train_frac)
        train_subs = set(subject_keys[:split_point])

        for sub, vids in subjects.items():
            split = "train" if sub in train_subs else "val"
            for v in vids:
                process_item(v, split, cls)

    return output_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames and audio from a dataset and split by subject")
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    print("Split dataset by subject and process:")
    subject_split_and_process(args.input, args.output, train_frac=args.train_frac, fps=args.fps, sr=args.sr)

