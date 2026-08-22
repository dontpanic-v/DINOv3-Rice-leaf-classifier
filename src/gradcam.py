import torch


class GradCAM:
    """Grad-CAM over a single ViT block via forward/backward hooks.

    Use as a context manager so hooks are always released.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=1, keepdim=True)
        cam = (weights * self.activations).sum(dim=-1)
        cam = torch.relu(cam)

        num_patches = cam.shape[1] - self.model.num_prefix_tokens
        side = int(num_patches ** 0.5)
        cam = cam[:, self.model.num_prefix_tokens:]
        cam = cam.reshape(1, side, side)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.squeeze().cpu().numpy(), target_class
