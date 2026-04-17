import os, shutil, random
import numpy as np
from glob import glob
from preprocess import extract_frames, extract_audio

def subject_split_and_process(root_videos, output_data, train_frac=0.8, fps=2, sr=16000):
    """
    Prepare dataset with subject-level splitting to prevent data leakage.
    Extracts frames and audio from videos, organizes into train/val splits.
    """
    class_paths = {
        'real': os.path.join(root_videos, 'RealVideo-RealAudio'),
        'fake': os.path.join(root_videos, 'FakeVideo-RealAudio'),
    }

    subject_dirs = {'real': [], 'fake': []}
    for label, path in class_paths.items():
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist")
            continue
        for root, dirs, files in os.walk(path):
            for d in dirs:
                if d.startswith('id'):
                    subject_dirs[label].append(os.path.join(root, d))

    print('Subjects found:', {k: len(v) for k, v in subject_dirs.items()})

    for label in ['real', 'fake']:
        random.seed(42)
        ids = subject_dirs[label]
        random.shuffle(ids)
        split = int(len(ids) * train_frac)
        train_ids = ids[:split]
        val_ids = ids[split:]
        
        for split_name, split_ids in [('train', train_ids), ('val', val_ids)]:
            fdir = os.path.join(output_data, split_name, 'frames', label)
            adir = os.path.join(output_data, split_name, 'audio', label)
            os.makedirs(fdir, exist_ok=True)
            os.makedirs(adir, exist_ok=True)
            
            for subject_dir in split_ids:
                subject_id = os.path.basename(subject_dir)
                subject_fdir = os.path.join(fdir, subject_id)
                subject_adir = os.path.join(adir, subject_id)
                os.makedirs(subject_fdir, exist_ok=True)
                os.makedirs(subject_adir, exist_ok=True)
                
                # Find video files in subject directory
                video_files = glob(os.path.join(subject_dir, '*.mp4')) + glob(os.path.join(subject_dir, '*.avi'))
                
                for video_path in video_files:
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    
                    # Extract frames
                    frame_dir = os.path.join(subject_fdir, video_name)
                    extract_frames(video_path, frame_dir, fps)
                    
                    # Extract audio
                    audio_path = os.path.join(subject_adir, f"{video_name}.wav")
                    extract_audio(video_path, audio_path, sr)

    print('Dataset prepared with frames and audio extracted.')
