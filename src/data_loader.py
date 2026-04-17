import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import librosa


class MultimodalDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None, n_frames=8, sr=16000, n_mels=128, duration=1.5):
        self.examples = []
        self.transform = transform
        self.n_frames = n_frames
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration

        for label, cls in [(0, "real"), (1, "fake")]:
            frame_base = os.path.join(data_root, split, "frames", cls)
            audio_base = os.path.join(data_root, split, "audio", cls)
            if not os.path.exists(frame_base) or not os.path.exists(audio_base):
                continue

            for subject_id in sorted(os.listdir(frame_base)):
                subject_dir = os.path.join(frame_base, subject_id)
                if not os.path.isdir(subject_dir):
                    continue

                for video_dir_name in sorted(os.listdir(subject_dir)):
                    video_dir = os.path.join(subject_dir, video_dir_name)
                    if not os.path.isdir(video_dir):
                        continue

                    audio_path = os.path.join(audio_base, subject_id, f"{video_dir_name}.wav")
                    if not os.path.exists(audio_path):
                        continue

                    self.examples.append((video_dir, audio_path, label))

        if len(self.examples) == 0:
            raise ValueError(f"No multimodal examples found under {data_root}/{split}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        frame_dir, audio_path, label = self.examples[idx]

        frame_paths = sorted(glob(os.path.join(frame_dir, "*.jpg")))
        if len(frame_paths) == 0:
            raise ValueError(f"No frames found in {frame_dir}")

        if len(frame_paths) >= self.n_frames:
            selected_indices = np.linspace(0, len(frame_paths) - 1, self.n_frames, dtype=int)
        else:
            selected_indices = list(range(len(frame_paths))) + [len(frame_paths) - 1] * (self.n_frames - len(frame_paths))

        frames = []
        for index in selected_indices:
            img = Image.open(frame_paths[index]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frame_tensor = torch.stack(frames)

        y, _ = librosa.load(audio_path, sr=self.sr)
        target_length = int(self.sr * self.duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]

        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, n_fft=1024, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        if mel_db.shape[1] < 128:
            mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode="constant", constant_values=np.min(mel_db))
        else:
            mel_db = mel_db[:, :128]

        mel_db = torch.tensor(mel_db).unsqueeze(0)
        return frame_tensor, mel_db, torch.tensor(label, dtype=torch.long)


def build_dataset(data_root, split="train", batch_size=16, num_workers=4, n_frames=8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = MultimodalDataset(data_root, split, transform=transform, n_frames=n_frames)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers, drop_last=False)
    return loader


def build_evaluation_loaders(data_root, batch_size=16, num_workers=4, n_frames=8):
    return {
        "train": build_dataset(data_root, "train", batch_size, num_workers, n_frames),
        "val": build_dataset(data_root, "val", batch_size, num_workers, n_frames),
    }

