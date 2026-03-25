# Multimodal Deepfake Detection (Capstone)

## Overview
This project implements a lightweight multimodal deepfake detector using video frames + audio. It includes:
- frame + audio extraction (preprocess)
- lightweight CNNs (MobileNetV2 for video, small conv stack for audio)
- fusion model for combined classification
- explainability (Grad-CAM for video)
- real-time demo with webcam overlay

## Install
```
pip install torch torchvision torchaudio opencv-python librosa scikit-learn numpy matplotlib pillow
```

## Directory structure
- `data/frames/real` and `data/frames/fake`
- `data/audio/real` and `data/audio/fake`
- `outputs/models`

## Run
1. Prepare raw dataset: `data/raw/real/*.mp4`, `data/raw/fake/*.mp4`.
2. Split and extract by subject: `python src/preprocess.py --input data/raw --output data --train_frac 0.8 --fps 2`.
3. Train: `python src/train.py --frame_root data --audio_root data --epochs 10 --batch_size 16`.
4. Eval: `python src/evaluate.py --checkpoint outputs/models/fusion_best.pth --split val`.
5. Realtime camera: `python src/realtime.py`.

## Notes
- Ensure `ffmpeg` is installed and in PATH for audio extraction.
- Use subject-level splits and cross-dataset generalization for research rigour.
