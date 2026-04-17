import os
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import build_evaluation_loaders
from models import FusionModel


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for frame_batch, audio_batch, labels in loader:
            frame_batch = frame_batch.to(device)
            audio_batch = audio_batch.to(device)
            labels = labels.to(device)
            outputs, _, _ = model(frame_batch, audio_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(emb_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    loaders = build_evaluation_loaders(cfg["frame_root"], cfg["batch_size"], cfg["num_workers"], cfg["n_frames"])

    best_val = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for frame_batch, audio_batch, labels in loaders["train"]:
            frame_batch = frame_batch.to(device)
            audio_batch = audio_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, _, _ = model(frame_batch, audio_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step()
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch}/{cfg['epochs']}  Loss={total_loss/total_samples:.4f}  Acc={train_acc:.4f}  LR={scheduler.get_last_lr()[0]:.2e}")

        if epoch % cfg["validate_every"] == 0:
            val_acc = evaluate(model, loaders["val"], device)
            print(f"Validation acc={val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                os.makedirs(cfg["save_dir"], exist_ok=True)
                ckpt_path = os.path.join(cfg["save_dir"], "fusion_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved best checkpoint: {ckpt_path}")

    print("Training completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train multimodal deepfake model")
    parser.add_argument("--frame_root", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="outputs/models")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--validate_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_frames", type=int, default=8)
    args = parser.parse_args()

    config = vars(args)
    train(config)
