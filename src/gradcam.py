"""Grad-CAM for ResNet18.

Hooks the last convolutional block (layer4), weights its feature maps by the
gradients for a target class, and returns a heatmap in [0, 1] plus an overlay.
"""
import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        # Last convolutional block of ResNet18.
        self.target_layer = target_layer if target_layer is not None else model.layer4
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        # register_full_backward_hook is the non-deprecated API for modules.
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        """Return (heatmap, predicted_class).

        input_tensor: (1, 3, 224, 224) tensor, already normalised.
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())
        score = logits[0, target_class]
        score.backward()

        # Global-average-pool the gradients -> one weight per feature map.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, target_class


def overlay_heatmap(pil_image, heatmap, alpha=0.5):
    """Overlay a Grad-CAM heatmap on the original PIL image.

    Returns a PIL image. Uses matplotlib's jet colormap, without importing
    matplotlib at module level in gradcam.py callers that do not need it.
    """
    import matplotlib.cm as cm

    image = pil_image.convert("RGB").resize((224, 224))
    heat = Image.fromarray((heatmap * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)
    coloured = cm.jet(np.array(heat) / 255.0)[:, :, :3]  # RGB, [0, 1]
    coloured = Image.fromarray((coloured * 255).astype("uint8"))
    return Image.blend(image, coloured, alpha)
