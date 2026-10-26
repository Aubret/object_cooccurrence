import math
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image



def fovea_mask(size):
    xs = torch.linspace(-1, 1, size)
    ys = torch.linspace(-1, 1, size)

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij"
    )

    r = torch.sqrt(xx**2 + yy**2)
    return r <= 1




def retinal_downsample(
    H,
    W,
    fovea_size = 32,
    output_size=112,
):
    """
    img: (1,C,H,W)
    fixation: (x,y) in pixels
    """

    # _, H, W = img.shape
    # x0, y0 = fixation
    ratio = H/output_size

# fovea_mask = fovea_mask(output_size)
    limit = 1 # / math.sqrt(2)
    xs = torch.linspace(-limit, limit, output_size)
    ys = torch.linspace(-limit, limit, output_size)
    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij"
    )

    r = torch.sqrt(xx**2 + yy**2)


    rfov = torch.zeros((output_size, output_size))
    rfov[output_size//2-fovea_size//2: output_size//2+fovea_size//2, output_size//2-fovea_size//2:output_size//2+fovea_size//2] = fovea_mask(fovea_size)

    r_img_fov = rfov / ratio # + r_img_periph * (1-rfov)
    xx_img = xx * r_img_fov
    yy_img = yy * r_img_fov

    d_fov = torch.max(r_img_fov)
    d_fov_in = torch.max(r * r_img_fov / (r_img_fov + 1e-7))

    # alpha_computed = torch.log(1 - d_fov) / (torch.log(torch.relu(1 - d_fov_in)) + 1e-6)
    # r_img_periph = d_fov + torch.relu(r - d_fov_in) ** alpha_computed

    alpha_computed = torch.log(1 - d_fov +1 ) / (torch.relu(1 - d_fov_in) + 1e-6)
    r_img_periph = d_fov + torch.exp(alpha_computed * (r - d_fov_in)) - 1
    print(alpha_computed)
    xx_img += xx * (1-rfov) * r_img_periph
    yy_img += yy * (1-rfov) * r_img_periph

    valid = (xx_img.abs() <= 1) & (yy_img.abs() <= 1)
    H = valid.shape[0]
    c = H // 2

    half = 0
    while True:
        if c - half < 0 or c + half >= H or not valid[c - half:c + half + 1, c - half:c + half + 1].all():
            break
        half += 1
    half -= 1
    crop_size = 2 * half + 1

    grid = torch.stack(
        [xx_img, yy_img],
        dim=-1
    )
    return grid, crop_size
    # print(xx_img)


class CorticalMagnificationDown:

    def __init__(self, H, W, fovea_size=32, output_size=112):
        self.grid, self.crop_size = retinal_downsample(H, W, fovea_size, output_size)


    def __call__(self, img):
        retinal = F.grid_sample(
            img.unsqueeze(0),
            self.grid.unsqueeze(0),
            align_corners=True,
        )

        retinal = torchvision.transforms.functional.center_crop(
            retinal,
            [self.crop_size, self.crop_size],
        )
        return retinal.squeeze(0)


if __name__ == "__main__":
    img_cm = Image.open("/home/arthur/postdoc/results/nymeria_images/0_1.png").convert("RGB")
    fov = 2
    K=0.1
    slope = 3
    img_cm = torchvision.transforms.functional.to_tensor(img_cm)
    # for alpha in [ 1 ]:
    for output_size in [64, 92, 128, 164, 196 ]:
        _, H, W = img_cm.shape
        grid, crop_size = retinal_downsample(H, W , fovea_size = 48, output_size=output_size)
        img_cm2 = F.grid_sample(
            img_cm.unsqueeze(0),
            grid.unsqueeze(0),
            align_corners=True,
        )
        torchvision.utils.save_image(img_cm2, f"/home/arthur/postdoc/results/cm/fovations{output_size}.png")
        # crop_size = int(output_size / math.sqrt(2))
        # crop_size = int(output_size / 1.2)
        retinal = torchvision.transforms.functional.center_crop(
            img_cm2,
            [crop_size, crop_size],
        )

        # torchvision.utils.save_image(img_cm, f"/home/arthur/postdoc/results/test_cm{fov}_{K}.png", )
        torchvision.utils.save_image(retinal.squeeze(), f"/home/arthur/postdoc/results/cm/fovations_cropped{output_size}.png")