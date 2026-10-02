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