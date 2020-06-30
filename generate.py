import argparse
import math
import time

import torch
from torchvision import utils
import numpy as np

from model import StyledGenerator

@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

@torch.no_grad()
def sample(generator, latents, num_layers, mean_style, latent_space_type):
    image = generator(
        latents,
        num_layers=num_layers,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
        latent_space_type=latent_space_type
    )

    return image

@torch.no_grad()
def style_mixing(generator, num_layers, mean_style, n_source, n_target, device):
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)

    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]

    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)

    return images

def slerp_1d(high_tensor, low_tensor, n_sample):
    high = high_tensor.numpy()
    low = low_tensor.numpy()
    assert high.ndim == 1
    assert low.ndim == 1

    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    slerps = []
    for i in range(n_sample):
        value = i*1.0/n_sample
        if so == 0:
            tensor = (1.0 - value) * low + value * high
            s = torch.tensor(tensor)  # L'Hopital's rule/LERP
            slerps.append(s)
        else:
            tensor = np.sin((1.0 - value) * omega) / so * low + np.sin(value * omega) / so * high
            s = torch.tensor(tensor)
            slerps.append(s)
    
    return torch.stack(slerps, 0)
        
def slerp(high_tensor, low_tensor, n_sample):
    slerps = []
    for high, low in zip(high_tensor, low_tensor):
        s = slerp_1d(high, low, n_sample)
        slerps.append(s)
    num_layers = len(slerps)
    slerps = torch.transpose(torch.stack(slerps, 0), 0, 1)
    for slerp in slerps:
        yield slerp

def lerp(z1, z0, n_sample):
    for i in range(n_sample):
        yield z0 + (z1 - z0) * i / n_sample

def interpolate(z0, z1, n_sample, func):
    latents = []
    for latent in func(z1, z0, n_sample):
        latents.append(latent)

    return torch.stack(latents, 0)

def get_latents(args):
    latent_space_type = args.latent_space_type
    num_layers = (int(math.log(args.size, 2)) - 1) * 2
    if args.latent_path and latent_space_type == 'wp':
        latents_1 = torch.from_numpy(np.load(args.latent_path))
        latents_2 = torch.from_numpy(np.load(args.other_latent_path))
        latents_1 = latents_1.reshape(-1, 512)
        latents_2 = latents_1.reshape(-1, 512)
    elif args.latent_path and (latent_space_type == 'z' or latent_space_type == 'w'):
        latents_1 = torch.from_numpy(np.load(args.latent_path))
        latents_2 = torch.from_numpy(np.load(args.other_latent_path))
        latents_1 = latents_1.reshape(-1, 512).repeat(num_layers, 1)
        latents_2 = latents_1.reshape(-1, 512).repeat(num_layers, 1)
    elif latent_space_type == 'z' or latent_space_type == 'w':
        latents_1, latents_2 = torch.randn(1, 512), torch.randn(1, 512)
        latents_1, latents_2 = latents_1.repeat(num_layers, 1), latents_2.repeat(num_layers, 1)
    elif latent_space_type == 'wp':
        latents_1, latents_2 = torch.randn(num_layers, 512), torch.randn(num_layers, 512)
    else:
        raise Exception("Latents space type not found")
    
    print(f"latents 1 shape: {latents_1.shape}")
    print(f"latents 2 shape: {latents_2.shape}")
    return latents_1, latents_2
    
def main(custom_args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024, help='size of the image')
    parser.add_argument('--n_row', type=int, default=3, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')

    parser.add_argument('--morph_interpolation', type=bool, default=False, help='whether to turn on style mixing mode, will return log(size, 2) images bounded by style 1 (image 0) and style 2 (image :-1)')
    parser.add_argument('--latent_path', type=str, default='', help='Path: latents of style 1')
    parser.add_argument('--other_latent_path', type=str, default='', help='Path: latents of style 2')
    parser.add_argument('--latent_space_type', type=str, default='z', help='latents type: z, w, wp')
    parser.add_argument('--morph_interpolation_type', type=str, default='lerp', help='lerp or slerp')
    parser.add_argument('--num_images', type=int, default=4, help='number of images in between 2 styles')

    parser.add_argument('--path', type=str, default='checkpoint/stylegan-512px-new.model', help='path to checkpoint')

    begin = time.time()

    if custom_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(custom_args)
 
    device = 'cuda'

    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(args.path)['g_running'])
    generator.eval()

    mean_style = get_mean_style(generator, device)

    num_layers = int(math.log(args.size, 2)) # 10
    n_sample = args.num_images + 2
    
    latents_1, latents_2 = get_latents(args)
    
    latents = interpolate(latents_1, latents_2, n_sample, slerp)
    latents = latents.to(device)
    print(f"Latents shape: {latents.shape}")
    imgs = sample(generator, latents, num_layers, mean_style, args.latent_space_type) #  args.n_row * args.n_col, device)
    
    print(imgs.shape)
    for j in range(n_sample):
        utils.save_image(imgs[j], f'sample_{j}.png', nrow=1 + 0 * args.n_col, normalize=True, range=(-1, 1))

#    utils.save_image(img, 'sample.png', nrow=1 + 0 * args.n_col, normalize=True, range=(-1, 1))
    print(f"Time duration: {time.time() - begin} secs")

#    for j in range(20):
#        img = style_mixing(generator, num_layers, mean_style, args.n_col, args.n_row, device)
#        utils.save_image(
#            img, f'sample_mixing_{j}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
#        )


if __name__ == '__main__':
    main()
    