import torch
import numpy as np
import cv2


def compute_gradcam(model, input_tensor, audio_tensor, target_class, layer_name="base.features"):
    model.train()
    model.audio_net.eval()
    model.fusion.eval()
    input_tensor.requires_grad = True
    outputs, _, _ = model(input_tensor, audio_tensor)
    loss = outputs[:, target_class].sum()
    with torch.backends.cudnn.flags(enabled=False):
        loss.backward()
    
    # Saliency map: absolute gradients of the input
    saliency = input_tensor.grad.abs().sum(dim=1, keepdim=True)  # Sum over channels
    saliency = saliency.squeeze(0).squeeze(0).cpu().numpy()  # (8, 224, 224) -> average over frames
    saliency = saliency.mean(axis=0)  # Average over frames
    
    # Normalize
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-8)
    
    model.eval()
    return saliency


def overlay_heatmap(frame, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1.0 - alpha, heatmap, alpha, 0)


def audio_mel_saliency(audio_tensor):
    """Simple saliency on mel-spectrogram: per-frequency band activation."""
    # audio_tensor: (B,1,M,T)
    if not isinstance(audio_tensor, torch.Tensor):
        raise TypeError("audio_tensor must be torch.Tensor")
    activation = audio_tensor.mean(dim=0).squeeze().cpu().numpy()
    saliency = (activation - np.min(activation)) / (np.ptp(activation) + 1e-8)
    return saliency


def mismatch_explanation(video_score, audio_score, threshold=0.2):
    """Produce human-readable explanation from modality score difference."""
    if abs(video_score - audio_score) < threshold:
        return "Modalities agree with high confidence; content likely consistent."

    dominant = "video" if video_score > audio_score else "audio"
    return (
        f"Detected mismatch: {dominant}-stream dominates (video={video_score:.2f}, audio={audio_score:.2f}). "
        "This may indicate deepfake manipulation in the weaker modality."
    )

