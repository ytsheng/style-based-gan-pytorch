import argparse
import math
import time

import torch
from torchvision import utils
import numpy as np

from model import StyledGenerator

from pyquaternion import Quaternion

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
def sample(generator, z, step, mean_style):
    image = generator(
        z,
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )

    return image

@torch.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target, device):
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

def slerp(high_tensor, low_tensor, n_sample):
    high = high_tensor.numpy()[0]
    low = low_tensor.numpy()[0]
    
    import ipdb; ipdb.set_trace()
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    for i in range(n_sample):
        value = i*1.0/n_sample
        if so == 0:
            tensor = (1.0 - value) * low + value * high
            yield torch.tensor(tensor).unsqueeze(0)  # L'Hopital's rule/LERP
        tensor = np.sin((1.0 - value) * omega) / so * low + np.sin(value * omega) / so * high
        yield torch.tensor(tensor).unsqueeze(0)

def lerp(z1, z0, n_sample):
    for i in range(n_sample):
        yield z0 + (z1 - z0) * i / n_sample

def interpolate(z0, z1, n_sample, func):
    latents = []
    for latent in func(z1, z0, n_sample):
        latents.append(latent)

    return torch.cat(latents, 0)

if __name__ == '__main__':
    begin = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024, help='size of the image')
    parser.add_argument('--n_row', type=int, default=3, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')
    parser.add_argument('path', type=str, help='path to checkpoint file')

    args = parser.parse_args()

    device = 'cuda'

    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(args.path)['g_running'])
    generator.eval()

    mean_style = get_mean_style(generator, device)

    step = int(math.log(args.size, 2)) - 2

    n_sample = 20
    z = interpolate(torch.randn(1, 512), torch.randn(1, 512), n_sample, slerp)
    z = z.to(device)
    imgs = sample(generator, z, step, mean_style) #  args.n_row * args.n_col, device)

    for j in range(n_sample):
        utils.save_image(imgs[j], f'sample_{j}.png', nrow=1 + 0 * args.n_col, normalize=True, range=(-1, 1))

#    utils.save_image(img, 'sample.png', nrow=1 + 0 * args.n_col, normalize=True, range=(-1, 1))
    print(f"Time duration: {time.time() - begin} secs")

#    for j in range(20):
#        img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device)
#        utils.save_image(
#            img, f'sample_mixing_{j}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
#        )
