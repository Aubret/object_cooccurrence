"""
Contrast-sensitivity simulation via frequency-domain thresholding
(Implementation based on the description in Lu et al., arXiv:2507.03168).
Citations: Lu et al., 2025. See arXiv:2507.03168 and ResearchGate copy for text description.
"""
import PIL
import numpy as np
import imageio
import os

import torchvision
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def _fft_power(img_gray):
    """Compute 2D FFT and power spectrum of a single-channel image"""
    F = fft2(img_gray)
    P = np.abs(F) ** 2
    return F, P

def _ifft_from_filtered(F_filtered):
    """Inverse FFT and return real image (clipped)"""
    img_rec = np.real(ifft2(F_filtered))
    return img_rec

def cs_threshold_from_params(max_power, cs_norm, beta=2e-4, lam=100, mapping='power'):
    """
    Map normalized contrast sensitivity cs_norm in [0,1] to a frequency-domain amplitude threshold.
    - max_power: maximum spectral power across RGB channels (scalar)
    - beta: base amplitude threshold hyperparameter (paper's beta)
    - lam: mapping factor (paper's lambda)
    - mapping: 'power' (default), 'exp', or 'linear'
    Returns scalar threshold (amplitude units).
    """
    if not (0.0 <= cs_norm <= 1.0):
        raise ValueError("cs_norm must be in [0,1]")

    if mapping == 'power':
        # threshold = beta * max_power * (1 - cs_norm) ** (1/lam)
        # chosen because lam from paper is large (50-150): exponent 1/lam is small.
        threshold = beta * max_power * ((1.0 - cs_norm) ** (1.0 / float(lam)))
    elif mapping == 'exp':
        # exponential decay mapping
        threshold = beta * max_power * np.exp(-float(lam) * cs_norm)
    elif mapping == 'linear':
        threshold = beta * max_power * (1.0 - cs_norm) / float(lam)
    else:
        raise ValueError("mapping must be 'power', 'exp', or 'linear'")

    return float(threshold)

import numpy as np
import cv2

def freq_filter_luminance(img, beta=2e-4, lam=100, cs_norm=0.5):

    # Image to float
    img = img.astype(np.float32) / 255.0

    # Convert RGB → YUV
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    Y, U, V = yuv[...,0], yuv[...,1], yuv[...,2]

    # FFT (luminance only)
    FY = np.fft.fftshift(np.fft.fft2(Y))
    PY = np.abs(FY)**2

    threshold = beta * PY.max() * ((1 - cs_norm)**(1/lam))
    mask = (PY >= threshold).astype(np.float32)

    # Filter only the luminance channel
    FYf = FY * mask
    Y_rec = np.real(np.fft.ifft2(np.fft.ifftshift(FYf)))

    # Reassemble, preserving chroma channels untouched
    YUV_rec = np.stack([Y_rec, U, V], axis=-1)
    RGB_rec = cv2.cvtColor(YUV_rec, cv2.COLOR_YUV2RGB)

    return np.clip(RGB_rec, 0, 1)


def apply_contrast_sensitivity(img, cs_norm, beta=2e-4, lam=100, mapping='power', thres=1, eps=1e-12):
    """
    img: HxWx3 uint8 or float image (RGB)
    cs_norm: contrast sensitivity normalized to [0,1] (0=newborn, 1=adult)
    beta, lam: hyperparameters from paper
    mapping: mapping type ('power', 'exp', 'linear')
    Returns contrast-limited image (float, same range as input).
    """
    # ensure float image in range [0,1]
    img_f = np.asarray(img).astype(np.float32)
    if img_f.max() > 1.0:
        img_f = img_f / 255.0

    H, W, C = img_f.shape
    assert C == 3, "expect RGB image"

    # compute FFT and power for each channel
    Fs = []
    Ps = []
    for ch in range(3):
        F, P = _fft_power(img_f[..., ch])
        Fs.append(F)
        Ps.append(P)
    Ps = np.stack(Ps, axis=0)  # shape (3,H,W)

    # maximum spectral power across RGB channels (scalar)
    max_power = float(Ps.max() + eps)

    # compute threshold
    # thresh = cs_threshold_from_params(max_power=max_power, cs_norm=cs_norm,
    #                                   beta=beta, lam=lam, mapping=mapping)

    # create mask: preserve frequency components with amplitude >= thresh
    # Note: we threshold amplitude (abs(F)), not power (abs(F)^2) to be closer to description;
    # if you prefer thresholding power, square thresh appropriately.
    # thresh = beta * max_power * (1.0 - cs_norm) # * 1/ max(t/lam, 1) #** (1.0 / lam))
    # thresh = beta * max_power * (1.0 - cs_norm) # * 1/ max(t/lam, 1) #** (1.0 / lam))

    P_all = np.maximum.reduce([Ps[0],
                               Ps[1],
                               Ps[2]])
    mask = (P_all >= thres).astype(np.float32)
    out_channels = np.zeros_like(img_f)
    # --- apply power-domain masking ---
    for ch in range(3):
        F = Fs[ch]
        # power = np.abs(F) ** 2
        # print(power)
        # mask = (power >= thresh).astype(np.float32)
        F_filtered = F * mask
        recon = _ifft_from_filtered(F_filtered)
        out_channels[..., ch] = recon

    # clip to [0,1]
    out = np.clip(out_channels, 0.0, 1.0)
    return out


def rgb_freq_filter(img, threshold_frac=0.2):
    """
    img: float32 RGB image in [0, 1]
    threshold_frac: percentage of highest frequencies to remove (0–1)
    """

    # Ensure float
    img = img.astype(np.float32)
    if img.max() > 1: img /= 255.0

    H, W, _ = img.shape

    # --- 1. Compute FFT for each channel ---
    F = []
    for c in range(3):
        Fc = np.fft.fftshift(np.fft.fft2(img[...,c]))
        F.append(Fc)

    # --- 2. Build *single radial mask* (identical for all channels) ---
    y, x = np.ogrid[:H, :W]
    cy, cx = H//2, W//2
    r = np.sqrt((y-cy)**2 + (x-cx)**2)
    print(r.min(), r.max())

    r_norm = (r - r.min())/(r.max() - r.min())
    print(r_norm.min(), r_norm.max())
    mask = (r_norm < threshold_frac).astype(np.float32)
    # mask shape = (H, W), identical for all channels

    # --- 3. Apply mask to all channels identically ---
    F_filtered = [Fc * mask for Fc in F]

    # --- 4. Inverse FFT ---
    out = []
    for Fc in F_filtered:
        channel = np.real(np.fft.ifft2(np.fft.ifftshift(Fc)))
        out.append(channel)

    out = np.stack(out, axis=-1)
    return np.clip(out, 0, 1)

def rgb_freq_filter_smooth(img, cutoff=0.3, sigma=0.1):
    """
    cutoff: center of transition (0–1)
    sigma: smoothness of the transition (0–1)
    """

    img = img.astype(np.float32)
    if img.max() > 1:
        img /= 255.0

    H, W, _ = img.shape

    # --- FFT per channel ---
    F = []
    for c in range(3):
        F.append(np.fft.fftshift(np.fft.fft2(img[...,c])))

    # --- Smooth radial mask ---
    y, x = np.ogrid[:H, :W]
    cy, cx = H//2, W//2
    r = np.sqrt((y-cy)**2 + (x-cx)**2)
    r_norm = r / r.max()

    # Raised-cosine mask (smooth low pass)
    mask = 0.5 * (1 + np.cos(np.pi * np.clip((r_norm - cutoff) / sigma, -1, 1)))
    mask[r_norm > cutoff + sigma] = 0
    mask[r_norm < cutoff - sigma] = 1

    # --- Apply mask to all channels ---
    Ff = [Fc * mask for Fc in F]

    # --- Inverse FFT ---
    out = []
    for Fc in Ff:
        out.append(np.real(np.fft.ifft2(np.fft.ifftshift(Fc))))

    out = np.stack(out, axis=-1)
    return np.clip(out, 0, 1)

# if __name__ == "__main__":
#     # small demo: load image, create low-cs and high-cs versions
#     import matplotlib.pyplot as plt
#
#     input_path = "/home/arthur/postdoc/projects/ATIPAVENIR26/apple_atipavenir.jpeg"   # replace with your image path
#     if not os.path.exists(input_path):
#         raise SystemExit(f"Place an RGB image named 'example.jpg' in this folder to run demo.")
#
#     img = imageio.v2.imread(input_path) / 255
#     print(img.shape, img.max(), img.min())
#     # Simulate newborn (cs_norm ~ 0.0) and adult (cs_norm ~ 1.0)
#     # newborn = apply_contrast_sensitivity(img, cs_norm=0.0, beta=2e-4, lam=100, thres=1e8, mapping='power')
#     # child = apply_contrast_sensitivity(img, cs_norm=0.9, beta=4e-4, lam=100, thres=10, mapping='power')
#     # adult = apply_contrast_sensitivity(img, cs_norm=1.0, beta=2e-4, lam=100, thres=2, mapping='power')
#     newborn = rgb_freq_filter_smooth(img, 0.05,0.01)
#     child = rgb_freq_filter_smooth(img, 0.5, 0.01)
#     adult = rgb_freq_filter_smooth(img, 1, 0.01)
#     # save outputs (uint8)
#     imageio.v2.imwrite("/home/arthur/postdoc/projects/ATIPAVENIR26/apple_newborn.jpeg", (newborn * 255).astype(np.uint8))
#     imageio.v2.imwrite("/home/arthur/postdoc/projects/ATIPAVENIR26/apple_child.jpeg", (child * 255).astype(np.uint8))
#     imageio.v2.imwrite("/home/arthur/postdoc/projects/ATIPAVENIR26/apple_adult.jpeg", (adult * 255).astype(np.uint8))
#
#     # quick visualization
#     fig, axs = plt.subplots(1,4, figsize=(12,4))
#     axs[0].imshow(img if img.max()>1.0 else img)
#     axs[0].set_title("original")
#     axs[1].imshow(newborn)
#     axs[1].set_title("newborn (cs~0)")
#     axs[2].imshow(child)
#     axs[2].set_title("child (cs~0.5)")
#     axs[3].imshow(adult)
#     axs[3].set_title("adult (cs~1.0)")
#     for a in axs:
#         a.axis('off')
#     plt.tight_layout()
#     plt.show()


if __name__ == "__main__":
    # small demo: load image, create low-cs and high-cs versions
    import matplotlib.pyplot as plt

    input_path = "/home/arthur/postdoc/projects/ATIPAVENIR26/apple_atipavenir.jpeg"   # replace with your image path
    if not os.path.exists(input_path):
        raise SystemExit(f"Place an RGB image named 'example.jpg' in this folder to run demo.")


    imgt = torchvision.transforms.functional.to_tensor(PIL.Image(input_path))
    print(img.shape, img.max(), img.min())
    # Simulate newborn (cs_norm ~ 0.0) and adult (cs_norm ~ 1.0)
    # newborn = apply_contrast_sensitivity(img, cs_norm=0.0, beta=2e-4, lam=100, thres=1e8, mapping='power')
    # child = apply_contrast_sensitivity(img, cs_norm=0.9, beta=4e-4, lam=100, thres=10, mapping='power')
    # adult = apply_contrast_sensitivity(img, cs_norm=1.0, beta=2e-4, lam=100, thres=2, mapping='power')
    newborn = rgb_freq_filter_smooth(img, 0.05,0.01)
    child = rgb_freq_filter_smooth(img, 0.5, 0.01)
    adult = rgb_freq_filter_smooth(img, 1, 0.01)
    # save outputs (uint8)
    imageio.v2.imwrite("/home/arthur/postdoc/projects/ATIPAVENIR26/apple_newborn.jpeg", (newborn * 255).astype(np.uint8))
    imageio.v2.imwrite("/home/arthur/postdoc/projects/ATIPAVENIR26/apple_child.jpeg", (child * 255).astype(np.uint8))
    imageio.v2.imwrite("/home/arthur/postdoc/projects/ATIPAVENIR26/apple_adult.jpeg", (adult * 255).astype(np.uint8))

    # quick visualization
    fig, axs = plt.subplots(1,4, figsize=(12,4))
    axs[0].imshow(img if img.max()>1.0 else img)
    axs[0].set_title("transform")
    for a in axs:
        a.axis('off')
    plt.tight_layout()
    plt.show()