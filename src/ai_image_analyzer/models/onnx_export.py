from __future__ import annotations
from typing import Optional


def export_clip_vision_encoder(
    model_name: str = "openai/clip-vit-base-patch32",
    out_path: str = "clip_vision.onnx",
    device: str = "cpu",
    opset: int = 14,
) -> tuple[str, float]:
    """
    Export CLIP vision encoder to ONNX format.
    Returns (path, size_mb).
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    class VisionEncoder(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.vision_model = clip_model.vision_model

        def forward(self, pixel_values):
            outputs = self.vision_model(pixel_values=pixel_values)
            # Use pooled output (CLS token) as image embedding
            return outputs.pooler_output

    wrapper = VisionEncoder(model).to(device).eval()

    # Dummy input
    dummy = torch.zeros(1, 3, 224, 224, device=device)

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["pixel_values"],
        output_names=["image_embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embedding": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    # Verify with ONNX Runtime
    import onnxruntime as ort
    session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"pixel_values": dummy.cpu().numpy()}
    ort_outs = session.run(None, ort_inputs)
    assert ort_outs[0].shape == (1, 512), f"Unexpected output shape: {ort_outs[0].shape}"

    size_mb = __import__("os").path.getsize(out_path) / (1024 * 1024)
    return out_path, size_mb

def export_detector_onnx(
    checkpoint_path: str,
    out_path: str,
    metadata: Optional[dict] = None,
    opset: int = 17,
) -> tuple[str, float, float]:
    """Export a trained real-vs-AI classifier (.pt from scripts/train.py) to a single ONNX file.

    The graph maps a normalized float32 NCHW batch to softmax probabilities [P(real), P(ai)].
    Preprocessing is NOT in the graph (see OnnxBinaryDetector). A JSON sidecar with the
    preprocessing contract and metrics is written next to the model.
    Returns (path, size_mb, max_abs_diff_vs_torch).
    """
    import json, os
    import numpy as np
    import timm
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch, size = ckpt.get("model_name", "efficientnet_b0"), int(ckpt.get("img_size", 192))
    net = timm.create_model(arch, pretrained=False, num_classes=int(ckpt.get("num_classes", 2)))
    net.load_state_dict(ckpt["model_state_dict"])

    # Class-activation decomposition. The net is  features -> global average pool -> Linear, so
    # logit_ai - logit_real = sum_{h,w} evidence[h,w] + bias_diff  holds EXACTLY (no gradients/occlusion needed).
    w_diff = (net.get_classifier().weight[1] - net.get_classifier().weight[0]).detach()
    bias_diff = float((net.get_classifier().bias[1] - net.get_classifier().bias[0]).detach())

    class Probs(torch.nn.Module):
        def __init__(self, m, w):
            super().__init__()
            self.m = m
            self.register_buffer("w", w)

        def forward(self, x):
            f = self.m.forward_features(x)                              # B,C,h,w
            logits = self.m.get_classifier()(self.m.forward_head(f, pre_logits=True))
            evidence = torch.einsum("bchw,c->bhw", f, self.w) / (f.shape[2] * f.shape[3])
            return torch.softmax(logits, dim=1), evidence

    wrapper = Probs(net, w_diff).eval()
    dummy = torch.randn(2, 3, size, size)
    torch.onnx.export(
        wrapper, dummy, out_path, input_names=["pixel_values"], output_names=["probabilities", "evidence_map"],
        dynamic_axes={"pixel_values": {0: "batch"}, "probabilities": {0: "batch"}, "evidence_map": {0: "batch"}},
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, size, size)
    with torch.no_grad():
        ref_p, ref_e = (t.numpy() for t in wrapper(x))
        logit = net(x)
    got_p, got_e = sess.run(None, {"pixel_values": x.numpy()})
    diff = float(max(np.abs(ref_p - got_p).max(), np.abs(ref_e - got_e).max()))
    assert diff < 1e-3, f"ONNX output deviates from torch by {diff}"
    # exactness of the decomposition: sum(evidence) + bias == logit_ai - logit_real
    recon = got_e.sum(axis=(1, 2)) + bias_diff
    true = (logit[:, 1] - logit[:, 0]).detach().numpy()
    assert np.abs(recon - true).max() < 1e-3, f"evidence map does not sum to logit diff: {np.abs(recon - true).max()}"

    meta = {
        "name": "detector_v2", "architecture": arch, "input_size": size,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "preprocess": ckpt.get("preprocess", "resize"),
        "classes": ["real", "ai"],
        "evidence_bias": bias_diff,
        "evidence_map": "sum(evidence_map) + evidence_bias == logit_ai - logit_real (exact class-activation decomposition)",
        **(metadata or {}),
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    return out_path, os.path.getsize(out_path) / (1024 * 1024), diff



def export_classifier_onnx(
    checkpoint_path: str,
    out_path: str,
    classes: list[str],
    metadata: Optional[dict] = None,
    opset: int = 17,
) -> tuple[str, float, float]:
    """Export a multi-class timm classifier (e.g. generator attribution) to ONNX: NCHW batch -> softmax probabilities."""
    import json, os
    import numpy as np
    import timm
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch, size = ckpt.get("model_name", "efficientnet_b0"), int(ckpt.get("img_size", 192))
    net = timm.create_model(arch, pretrained=False, num_classes=len(classes))
    net.load_state_dict(ckpt["model_state_dict"])

    class Probs(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.softmax(self.m(x), dim=1)

    wrapper = Probs(net).eval()
    torch.onnx.export(wrapper, torch.randn(2, 3, size, size), out_path, input_names=["pixel_values"],
                      output_names=["probabilities"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "probabilities": {0: "batch"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, size, size)
    with torch.no_grad():
        ref = wrapper(x).numpy()
    diff = float(np.abs(ref - sess.run(None, {"pixel_values": x.numpy()})[0]).max())
    assert diff < 1e-3, f"ONNX output deviates from torch by {diff}"
    meta = {"architecture": arch, "input_size": size, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "preprocess": ckpt.get("preprocess", "center_crop_256_jpeg95"), "classes": list(classes), **(metadata or {})}
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    return out_path, os.path.getsize(out_path) / (1024 * 1024), diff
