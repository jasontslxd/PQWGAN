import os
import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image

from utils.dataset import load_mnist, load_fmnist, denorm, select_from_dataset
from utils.wgan import compute_gradient_penalty
from models.QGCC import PQWGAN_CC
from models. QGQC import PQWGAN_QC

def train(classes_str, dataset_str, patches, layers, n_data_qubits, batch_size, out_folder, checkpoint, randn, patch_shape, qcritic):
    classes = list(set([int(digit) for digit in classes_str]))

    device = torch.device("cpu")
    n_epochs = 50
    image_size = 28
    channels = 1
    if dataset_str == "mnist":
        dataset = select_from_dataset(load_mnist(image_size=image_size), 1000, classes)
    elif dataset_str == "fmnist":
        dataset = select_from_dataset(load_fmnist(image_size=image_size), 1000, classes)

    ancillas = 1
    if n_data_qubits:
        qubits = n_data_qubits + ancillas
    else:
        qubits = math.ceil(math.log(image_size ** 2 // patches, 2)) + ancillas

    if qcritic:
        lr_D = 0.01
    else:
        lr_D = 0.0002
    lr_G = 0.01
    b1 = 0
    b2 = 0.9
    latent_dim = qubits
    lambda_gp = 10
    n_critic = 5
    sample_interval = 10
    out_dir = f"{out_folder}/{classes_str}_{patches}p_{layers}l_{batch_size}bs"
    if randn:
        out_dir += "_randn"
    if patch_shape[0] and patch_shape[1]:
        out_dir += f"_{patch_shape[0]}x{patch_shape[1]}ps"
    
    os.makedirs(out_dir,exist_ok=True)
    if qcritic:
        gan = PQWGAN_QC(image_size=image_size, channels=channels, n_generators=patches, n_gen_qubits=qubits, n_ancillas=ancillas, n_gen_layers=layers, patch_shape=patch_shape, n_critic_qubits=10, n_critic_layers=175)
    else:
        gan = PQWGAN_CC(image_size=image_size, channels=channels, n_generators=patches, n_qubits=qubits, n_ancillas=ancillas, n_layers=layers, patch_shape=patch_shape)
    critic = gan.critic.to(device)
    generator = gan.generator.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    optimizer_G = Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))
    optimizer_C = Adam(critic.parameters(), lr=lr_D, betas=(b1, b2))
    if randn:
        fixed_z = torch.randn(batch_size, latent_dim, device=device)
    else:
        fixed_z = torch.rand(batch_size, latent_dim, device=device)

    wasserstein_distance_history = []
    saved_initial = False
    batches_done = 0

    if checkpoint != 0:
        critic.load_state_dict(torch.load(out_dir + f"/critic-{checkpoint}.pt"))
        generator.load_state_dict(torch.load(out_dir + f"/generator-{checkpoint}.pt"))
        wasserstein_distance_history = list(np.load(out_dir + "/wasserstein_distance.npy"))
        saved_initial = True
        batches_done = checkpoint

    for epoch in range(n_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            if not saved_initial:
                fixed_images = generator(fixed_z)
                save_image(denorm(fixed_images), os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
                save_image(denorm(real_images), os.path.join(out_dir, 'real_samples.png'), nrow=5)
                saved_initial = True

            real_images = real_images.to(device)
            optimizer_C.zero_grad()

            if randn:
                z = torch.randn(batch_size, latent_dim, device=device)
            else:
                z = torch.rand(batch_size, latent_dim, device=device)
            fake_images = generator(z)

            # Real images
            real_validity = critic(real_images)
            # Fake images
            fake_validity = critic(fake_images)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(critic, real_images, fake_images, device)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)
            wasserstein_distance_history.append(wasserstein_distance.item())

            d_loss.backward()
            optimizer_C.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_images = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = critic(fake_images)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Wasserstein Distance: {wasserstein_distance.item()}]")
                np.save(os.path.join(out_dir, 'wasserstein_distance.npy'), wasserstein_distance_history)
                batches_done += n_critic

                if batches_done % sample_interval == 0:
                    fixed_images = generator(fixed_z)
                    save_image(denorm(fixed_images), os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
                    torch.save(critic.state_dict(), os.path.join(out_dir, 'critic-{}.pt'.format(batches_done)))
                    torch.save(generator.state_dict(), os.path.join(out_dir, 'generator-{}.pt'.format(batches_done)))
                    print("saved images and state")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cl", "--classes", help="classes to train on", type=str)
    parser.add_argument("-d", "--dataset", help="dataset to train on", type=str)
    parser.add_argument("-p", "--patches", help="number of sub-generators", type=int, choices=[1,2,4,7,14,28])
    parser.add_argument("-l", "--layers", help="layers per sub-generators", type=int)
    parser.add_argument("-q", "--qubits", help="number of data qubits per sub-generator", type=int, default=None)
    parser.add_argument("-b", "--batch_size", help="batch_size", type=int)
    parser.add_argument("-o", "--out_folder", help="output directory", type=str)
    parser.add_argument("-c", "--checkpoint", help="checkpoint to load from", type=int, default=0)
    parser.add_argument("-rn", "--randn", help="use normal prior, otherwise use uniform prior", action="store_true")
    parser.add_argument("-ps", "--patch_shape", help="shape of sub-generator output (H, W)", default=[None,None], type=int, nargs=2)
    parser.add_argument("-qc", "--qcritic", help="use quantum critic", action="store_true")
    args = parser.parse_args()
    
    train(args.classes, args.dataset, args.patches, args.layers, args.qubits, args.batch_size, args.out_folder, args.checkpoint, args.randn, tuple(args.patch_shape), args.qcritic)