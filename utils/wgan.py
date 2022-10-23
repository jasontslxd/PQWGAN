import torch
import torch.autograd as autograd

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    batch_size, C, W, H = real_samples.shape
    epsilon = torch.rand(batch_size, 1, 1, 1).repeat(1, C, W, H).to(device)
    interpolated_images = (epsilon * real_samples + ((1 - epsilon) * fake_samples))
    interpolated_scores = critic(interpolated_images)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        inputs=interpolated_images,
        outputs=interpolated_scores,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = torch.mean((1. - torch.sqrt(1e-8+torch.sum(gradients**2, dim=1)))**2)
    return gradient_penalty