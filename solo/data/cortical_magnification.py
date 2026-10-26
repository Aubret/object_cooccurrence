from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pylab as plt
import matplotlib as mpl
import sys
sys.path.append("../..")

from solo.data.foveation import foveation

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
from scipy.stats import norm



def unravel_indices(
  indices: torch.LongTensor,
  shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.
    Args:
    indices: A tensor of (flat) indices, (*, N).
    shape: The targeted shape, (D,).
    Returns:
    The unraveled coordinates, (*, N, D).
    """
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        # if OLD_DIV:
        #     indices = indices // dim
        # else:
            # use this version to suppress the torch warning about changed behavior in newer torch.
        indices = torch.div(indices, dim, rounding_mode="floor")
    coord = torch.stack(coord[::-1], dim=-1)
    return coord


def get_RandomMagnifTfm(grid_generator="radial_quad_isotrop", bdr=16, fov=20, K=20, slope_C=0.012,
                        sal_sample=False, sample_temperature=1.5, **kwargs):
    """Constructor of a random Magnification Transform with signature
        randomMagnif(imgtsr, logdensity=None)

    Parameters:
        grid_generator: str

    """
    if grid_generator == "radial_quad_isotrop":
        grid_func = lambda imgtsr, pnt: radial_quad_isotrop_gridfun(imgtsr, pnt,
                                                                    fov=fov, K=K, **kwargs)
    elif grid_generator == "radial_exp_isotrop":
        grid_func = lambda imgtsr, pnt: radial_exp_isotrop_gridfun(imgtsr, pnt,
                                                                   slope_C=slope_C, **kwargs)
    elif grid_generator == "linear_separable":
        grid_func = lambda imgtsr, pnt: linear_separable_gridfun(imgtsr, pnt, **kwargs)
    elif grid_generator == "normal":
        grid_func = lambda imgtsr, pnt: normal_gridfun(imgtsr, pnt, **kwargs)
    else:
        raise NotImplementedError

    def randomMagnif(imgtsr, logdensity=None):
        _, H, W = imgtsr.shape
        if sal_sample and logdensity is not None:
            # print("sample with saliency")
            density = torch.exp((logdensity - logdensity.max()) / sample_temperature)
            # set the border density to be 0,
            density_mat = torch.zeros_like(density[0, 0, :, :])
            density_mat[bdr:H - bdr, bdr: W - bdr] = density[0, 0, bdr:H - bdr, bdr: W - bdr]
            # sample from density
            flat_idx = torch.multinomial(density_mat.flatten(), 1, replacement=True).cpu()
            cnt_coord = unravel_indices(flat_idx, density_mat.shape)
            pY, pX = cnt_coord[0, 0].item(), cnt_coord[0, 1].item()
        else:
            pY = np.random.randint(bdr, H - bdr)
            pX = np.random.randint(bdr, W - bdr)
        return img_cortical_magnif_tsr(imgtsr, (pY, pX), grid_func, demo=False)  # debug Oct.7th

    return randomMagnif


def img_cortical_magnif_tsr(imgtsr, pnt, grid_func, demo=False):
    if imgtsr.ndim == 4:
        imgtsr.squeeze_(0)
    _, H, W = imgtsr.shape
    XX_intp, YY_intp = grid_func(imgtsr, pnt)
    grid = torch.stack([(torch.tensor(XX_intp) / W * 2 - 1),  # normalize the value to -1, 1
                        (torch.tensor(YY_intp) / H * 2 - 1)],  # normalize the value to -1, 1
                       dim=2).unsqueeze(0).float()
    # print(grid.shape) # 1, H, W, 2
    img_cm = F.grid_sample(imgtsr.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    img_cm.squeeze_(0)
    if demo:
        # % Visualize the Manified plot.
        pY, pX = pnt
        figh, axs = plt.subplots(3, 1, figsize=(6, 12))
        axs[0].imshow(img_cm.permute([1, 2, 0]))
        axs[0].axis("off")
        axs[1].imshow(imgtsr.permute([1, 2, 0]))
        axs[1].axis("off")
        axs[1].scatter([pX], [pY], c='r', s=16, alpha=0.5)
        axs[2].scatter(XX_intp[::2, ::2].flatten(), YY_intp[::2, ::2].flatten(), c="r", s=0.25, alpha=0.2)
        axs[2].set_xlim([0, W])
        axs[2].set_ylim([0, H])
        axs[2].invert_yaxis()
        plt.show()
    return img_cm


def cortical_magnif_tsr_demo(imgtsr, pnt, grid_func, subN=2):
    """ Demo the cortical magnification transform.
    Inputs:
        imgtsr: Image tensor with shape (C, H, W) or (1, C, H, W)
        pnt: (X, Y) tuple.
        grid_func: a function with signature
            `XX_intp, YY_intp = grid_func(imgtsr, pnt)`
            It generates interpolation grid of X and Y.

    Parameters:
        subN: subsample the sampling grid by `subN` to show in the figure.
    """
    if imgtsr.ndim == 4:
        imgtsr.squeeze_(0)
    _, H, W = imgtsr.shape
    XX_intp, YY_intp = grid_func(imgtsr, pnt)
    grid = torch.stack([(torch.tensor(XX_intp) / W * 2 - 1),  # normalize the value to -1, 1
                        (torch.tensor(YY_intp) / H * 2 - 1)],  # normalize the value to -1, 1
                       dim=2).unsqueeze(0).float()
    img_cm = F.grid_sample(imgtsr.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros')
    img_cm.squeeze_(0)
    # % Visualize the Manified plot.
    pY, pX = pnt
    figh, axs = plt.subplots(3, 1, figsize=(6, 12))
    axs[0].imshow(img_cm.permute([1, 2, 0]))
    axs[0].axis("off")
    axs[1].imshow(imgtsr.permute([1, 2, 0]))
    axs[1].axis("off")
    axs[1].scatter([pX], [pY], c='r', s=16, alpha=0.5)
    axs[2].scatter(XX_intp[::subN, ::subN].flatten(),
                   YY_intp[::subN, ::subN].flatten(), c="r", s=0.25, alpha=0.2)
    axs[2].set_xlim([0, W])
    axs[2].set_ylim([0, H])
    axs[2].invert_yaxis()
    plt.show()
    return figh, img_cm, imgtsr


def linear_separable_gridfun(imgtsr, pnt, ):
    _, H, W = imgtsr.shape
    Hhalf, Whalf = H // 2, W // 2
    Hsum = Hhalf * (Hhalf + 1) / 2
    Wsum = Whalf * (Whalf + 1) / 2
    pY, pX = pnt
    UpDelta = pY / Hsum
    LeftDelta = pX / Wsum
    DownDelta = (H - pY) / Hsum
    RightDelta = (W - pX) / Wsum
    Left_ticks = np.cumsum(LeftDelta * np.arange(Whalf, 0, -1))
    Right_ticks = np.cumsum(RightDelta * np.arange(1, Whalf + 1, 1)) + pX
    Up_ticks = np.cumsum(UpDelta * np.arange(Hhalf, 0, -1))
    Down_ticks = np.cumsum(DownDelta * np.arange(1, Hhalf + 1, 1)) + pY
    X_ticks = np.hstack((Left_ticks, Right_ticks))
    Y_ticks = np.hstack((Up_ticks, Down_ticks))
    XX_intp, YY_intp = np.meshgrid(X_ticks, Y_ticks, )
    return XX_intp, YY_intp


def normal_gridfun(imgtsr, pnt, cutoff_std=2.25):
    """

    cutoff_std: where to cut off the normal distribution. too large will make the sampling at center
        too dense!
    """
    _, H, W = imgtsr.shape
    Hhalf, Whalf = H // 2, W // 2
    Hdensity = norm.pdf(np.linspace(0, cutoff_std, Hhalf))
    Wdensity = norm.pdf(np.linspace(0, cutoff_std, Whalf))
    H_delta = (1 / Hdensity)
    W_delta = (1 / Wdensity)
    Hsum = H_delta.sum()
    Wsum = W_delta.sum()
    pY, pX = pnt
    UpDelta = pY / Hsum
    LeftDelta = pX / Wsum
    DownDelta = (H - pY) / Hsum
    RightDelta = (W - pX) / Wsum
    Left_ticks = np.cumsum(LeftDelta * W_delta[::-1])
    Right_ticks = np.cumsum(RightDelta * W_delta[::]) + pX
    Up_ticks = np.cumsum(UpDelta * H_delta[::-1])
    Down_ticks = np.cumsum(DownDelta * H_delta[::]) + pY
    X_ticks = np.hstack((Left_ticks, Right_ticks))
    Y_ticks = np.hstack((Up_ticks, Down_ticks))
    XX_intp, YY_intp = np.meshgrid(X_ticks, Y_ticks, )
    return XX_intp, YY_intp


def radial_quad_isotrop_gridfun(imgtsr, pnt, fov=20, K=20, cover_ratio=None):
    _, H, W = imgtsr.shape
    Hhalf, Whalf = H // 2, W // 2
    pY, pX = pnt
    maxdist = np.sqrt(max(H - pY, pY) ** 2 + max(W - pX, pX) ** 2)  # in pixel
    grid_y, grid_x = np.mgrid[-Hhalf + 0.5:Hhalf + 0.5, -Whalf + 0.5:Whalf + 0.5]
    ecc2 = grid_y ** 2 + grid_x ** 2  # R2
    ecc = np.sqrt(ecc2)
    # RadDistTfm = lambda R, R2 : (R < fov) * R + (R > fov) * (R**2 - fov**2 + fov)
    RadDistTfm = lambda R: (R < fov) * R + \
                           (R > fov) * ((R + K) ** 2 / 2 / (fov + K) + fov - (fov + K) / 2)

    ecc_tfm = RadDistTfm(ecc, )
    coef = maxdist / ecc_tfm.max()
    if cover_ratio is not None:
        if type(cover_ratio) in [list, tuple]:
            ratio = np.random.uniform(cover_ratio[0], cover_ratio[1])
            coef = coef * np.sqrt(ratio)
        else:
            coef = coef * np.sqrt(cover_ratio)  # may not be optimal
    XX_intp = pX + coef * ecc_tfm * (grid_x / ecc)  # cosine
    YY_intp = pY + coef * ecc_tfm * (grid_y / ecc)  # sine
    return XX_intp, YY_intp


def radial_exp_isotrop_gridfun(imgtsr, pnt, slope_C=2.0, cover_ratio=None):
    _, H, W = imgtsr.shape
    Hhalf, Whalf = H // 2, W // 2
    pY, pX = pnt
    maxdist = np.sqrt(max(H - pY, pY) ** 2 + max(W - pX, pX) ** 2)  # in pixel
    grid_y, grid_x = np.mgrid[-Hhalf + 0.5:Hhalf + 0.5, -Whalf + 0.5:Whalf + 0.5]
    ecc2 = grid_y ** 2 + grid_x ** 2  # R2
    ecc = np.sqrt(ecc2)
    if type(slope_C) in [list, tuple]:
        slope = np.random.uniform(slope_C[0], slope_C[1])
    else:
        slope = slope_C  # may not be optimal
    RadDistTfm = lambda R: 1 / slope * (np.exp(slope * R / np.sqrt(Hhalf ** 2 + Whalf ** 2)) - 1)  # normalization
    ecc_tfm = RadDistTfm(ecc, )
    coef = maxdist / ecc_tfm.max()
    if cover_ratio is not None:
        if type(cover_ratio) in [list, tuple]:
            ratio = np.random.uniform(cover_ratio[0], cover_ratio[1])
            coef = coef * np.sqrt(ratio)
        else:
            coef = coef * np.sqrt(cover_ratio)  # may not be optimal
    XX_intp = pX + coef * ecc_tfm * (grid_x / ecc)  # cosine
    YY_intp = pY + coef * ecc_tfm * (grid_y / ecc)  # sine
    return XX_intp, YY_intp

class CorticalMagnification(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.fov = fov
        # self.K = K


    def forward(self, img):
        img = torchvision.transforms.functional.to_tensor(img)
        return img_cortical_magnif_tsr(img, (img.shape[1]//2, img.shape[2]//2),
                                      lambda im, pnt: radial_exp_isotrop_gridfun(im, pnt, slope_C=3),
                                      demo=False)
        # return img_cortical_magnif_tsr(img, (img.shape[1]//2, img.shape[2]//2),
        #                               lambda img2, pnt: radial_quad_isotrop_gridfun(img2, pnt, fov=self.fov,
        #                                                                             K=self.K), demo=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"



class CenterCropBig(torch.nn.Module):
    def __init__(self, size=None, ratio="1:1"):
        super().__init__()
        self.size = size
        self.ratio = ratio

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.size is None:
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size
            ratio = self.ratio.split(":")
            ratio = float(ratio[0]) / float(ratio[1])
            # Size must match the ratio while cropping to the edge of the image
            ratioed_w = int(h * ratio)
            ratioed_h = int(w / ratio)
            if w>=h:
                if ratioed_h <= h:
                    size = (ratioed_h, w)
                else:
                    size = (h, ratioed_w)
            else:
                if ratioed_w <= w:
                    size = (h, ratioed_w)
                else:
                    size = (ratioed_h, w)
        else:
            size = self.size
        return torchvision.transforms.functional.center_crop(img, size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

# %%
if __name__ == "__main__":
    # %%
    from scipy.datasets import face
    from skimage.transform import rescale

    # img = rescale(face(), (0.25, 0.25, 1))
    # imgtsr = torch.tensor(img).permute([2, 0, 1]).float()
    # # %%
    # img_cm = img_cortical_magnif_tsr(imgtsr, (80, 120), linear_separable_gridfun, demo=True)
    # # %%
    # img_cm = img_cortical_magnif_tsr(imgtsr, (80, 120), normal_gridfun, demo=True)
    # # %%
    # img_cm = img_cortical_magnif_tsr(imgtsr, (10, 190),
    #                                  lambda img, pnt: radial_quad_isotrop_gridfun(img, pnt, fov=20, K=20), demo=True)
    # # %%
    # img_cm = img_cortical_magnif_tsr(imgtsr, (100, 30),
    #                                  lambda img, pnt: radial_exp_isotrop_gridfun(img, pnt, slope_C=2.0,
    #                                                                              cover_ratio=0.4), demo=True)

    # imgtsr = torchvision.transforms.functional.to_tensor(Image.open("/home/aubret/Documents/postdoc/gym_results/test_images/ego4d_select/frames/output_frame_0558.png").convert("RGB"))
    #
    # imgtsr = Image.open("/home/arthur/postdoc/projects/ATIPAVENIR26/apple_atipavenir.jpeg").convert("L")
    # blur_radius = 10  # Adjust this value for more/less blur
    # blurred_img = imgtsr.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    #
    # contrast_factor = 0.3  # 0.5 = half the contrast
    # enhancer = ImageEnhance.Contrast(blurred_img)
    # blurred_img = enhancer.enhance(contrast_factor)
    #
    # img_cm= foveation(blurred_img, (700, 920), e_o=2)
    #
    # imgtsr = torchvision.transforms.functional.to_tensor(Image.open("/home/arthur/postdoc/projects/ATIPAVENIR26/apple_atipavenir.jpeg").convert("RGB"))
    # img_cm = img_cortical_magnif_tsr(img_cm, (600, 900),
    #                                  lambda img, pnt: radial_quad_isotrop_gridfun(img, pnt, fov=10, K=5), demo=False)
    # torchvision.utils.save_image(img_cm, "/home/arthur/postdoc/projects/ATIPAVENIR26/apple_atipavenir4.jpeg")

    ## Nature paper
    img_cm = Image.open("/home/arthur/postdoc/datasets/nymeria/tests224p/0.png").convert("RGB")
    fov = 2
    K=0.1
    slope = 3
    img_cm = torchvision.transforms.functional.to_tensor(img_cm)
    img_cm = img_cortical_magnif_tsr(img_cm, (112, 112),
                                     # lambda img, pnt: radial_quad_isotrop_gridfun(img, pnt, fov=fov, K=K), demo=False)
                                     lambda img, pnt: radial_exp_isotrop_gridfun(img, pnt, slope_C=slope), demo=False)
    # torchvision.utils.save_image(img_cm, f"/home/arthur/postdoc/results/test_cm{fov}_{K}.png", )
    torchvision.utils.save_image(img_cm, f"/home/arthur/postdoc/results/test_cm{slope}.png", )

    # %%  linear_separable
    # rndMagnif = get_RandomMagnifTfm(grid_generator="radial_quad_isotrop", bdr=16, fov=20, K=0, cover_ratio=(0.05, 1))
    # mtg = make_grid([rndMagnif(imgtsr) for _ in range(9)], nrow=3)
    # mtg_pil = ToPILImage()(mtg)
    # mtg_pil.show()
    #

    # %%
    # rndMagnif = get_RandomMagnifTfm(grid_generator="normal", bdr=16, fov=30, K=5)
    # mtg = make_grid([rndMagnif(imgtsr) for _ in range(9)], nrow=3)
    # mtg_pil = ToPILImage()(mtg)
    # mtg_pil.show()
    # # %%
    # rndMagnif = get_RandomMagnifTfm(grid_generator="radial_exp_isotrop", bdr=64, slope_C=(0.75, 3.0),
    #                                 cover_ratio=(0.1, 0.5))
    # mtg = make_grid([rndMagnif(imgtsr) for _ in range(9)], nrow=3)
    # mtg_pil = ToPILImage()(mtg)
    # mtg_pil.show()
