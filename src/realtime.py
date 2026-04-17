import cv2
import torch
import numpy as np
from models import FusionModel


def video_stream_inference(model, device, source=0, window_size=8, step=2):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Video source not opened: {source}. Please check if webcam is available or provide a video file path.")
        return

    frames_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_tensor = torch.tensor(frame_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        frames_buffer.append(frame_tensor)

        if len(frames_buffer) >= window_size:
            clip = torch.stack(frames_buffer[-window_size:]).unsqueeze(0).to(device)
            audio_input = torch.zeros((1, 1, 64, 64), device=device)

            with torch.no_grad():
                outputs, vemb, aemb = model(clip, audio_input)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                fake_score = float(probs[1])
                real_score = float(probs[0])
                label = "FAKE" if fake_score > 0.5 else "REAL"

            display = frame.copy()
            cv2.putText(display, f"{label}  fake={fake_score:.2f}  real={real_score:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if label == "REAL" else (0, 0, 255), 2)
            cv2.imshow("Real-Time Deepfake Detection", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            for _ in range(step):
                if frames_buffer:
                    frames_buffer.pop(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(emb_dim=128).to(device)
    model_path = "outputs/models/fusion_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    video_stream_inference(model, device, source=0)

