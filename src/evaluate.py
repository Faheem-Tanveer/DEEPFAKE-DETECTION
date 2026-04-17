import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

from data_loader import build_dataset
from models import FusionModel


def full_evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for frame_batch, audio_batch, labels in loader:
            frame_batch = frame_batch.to(device)
            audio_batch = audio_batch.to(device)
            labels = labels.to(device)

            outputs, _, _ = model(frame_batch, audio_batch)
            proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(proba.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_checkpoint(data_root, checkpoint, split="val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(emb_dim=128).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    loader = build_dataset(data_root, split, batch_size=32, num_workers=4)
    metrics = full_evaluate(model, loader, device)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate multimodal deepfake model")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    args = parser.parse_args()

    metrics = evaluate_checkpoint(args.data_root, args.checkpoint, args.split)
    print(metrics)

