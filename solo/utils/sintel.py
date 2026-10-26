import random
import torch
import torchvision
from torchvision.transforms import InterpolationMode

def sintel_transform(
    img1,
    img2,
    flow,
    valid_flow_mask=None,
    train=True,
    crop_size=(368, 768)
    # crop_size=(436, 1024)
):
    """
    img1, img2: PIL Images
    flow: numpy array [2, H, W]
    valid_flow_mask: numpy array [H, W] or None
    """

    # -------------------------
    # Convert to tensors
    # -------------------------
    img1 = torchvision.transforms.functional.to_tensor(img1)
    img2 = torchvision.transforms.functional.to_tensor(img2)

    flow = torch.from_numpy(flow).float() if flow is not None else None

    if valid_flow_mask is not None:
        valid_flow_mask = torch.from_numpy(valid_flow_mask).bool()
    else:
        H, W = img1.shape[1:]
        valid_flow_mask = torch.ones((1, H, W), dtype=torch.bool)

        # valid_flow_mask = torch.ones((1,*flow.shape[1:]), dtype=torch.bool)

    # -------------------------
    # Geometric augmentations
    # -------------------------
    if train:

        # ---- Random scaling ----
        scale = 2 ** random.uniform(-0.2, 0.4)  # ~[0.87, 1.32]
        H, W = img1.shape[1:]

        new_h = int(H * scale)
        new_w = int(W * scale)

        img1 = torchvision.transforms.functional.resize(img1, (new_h, new_w))
        img2 = torchvision.transforms.functional.resize(img2, (new_h, new_w))

        flow = torchvision.transforms.functional.resize(flow, (new_h, new_w))
        flow = flow * scale  # IMPORTANT

        valid_flow_mask = torchvision.transforms.functional.resize(
            valid_flow_mask.float(), (new_h, new_w), interpolation=InterpolationMode.NEAREST
        ).bool()

        # ---- Random crop ----
        ch, cw = crop_size
        _, H, W = img1.shape

        if H > ch and W > cw:
            y0 = random.randint(0, H - ch)
            x0 = random.randint(0, W - cw)

            img1 = img1[:, y0:y0 + ch, x0:x0 + cw]
            img2 = img2[:, y0:y0 + ch, x0:x0 + cw]
            flow = flow[:, y0:y0 + ch, x0:x0 + cw]
            valid_flow_mask = valid_flow_mask[:, y0:y0 + ch, x0:x0 + cw]
        # else:
        #     img1 = torchvision.transforms.functional.resize(img1, (new_h, new_w))
        #     img2 = torchvision.transforms.functional.resize(img2, (new_h, new_w))
        #     flow = torchvision.transforms.functional.resize(flow, (new_h, new_w))
        #     flow = flow * scale  # IMPORTANT
        #     valid_flow_mask = torchvision.transforms.functional.resize(
        #         valid_flow_mask.float(), (new_h, new_w), interpolation=InterpolationMode.NEAREST
        #     ).bool()

        # ---- Horizontal flip ----
        if random.random() < 0.5:
            img1 = torch.flip(img1, dims=[2])
            img2 = torch.flip(img2, dims=[2])
            flow = torch.flip(flow, dims=[2])
            valid_flow_mask = torch.flip(valid_flow_mask, dims=[2])

            flow[0] = -flow[0]  # invert x-flow

    # -------------------------
    # Photometric augmentations
    # -------------------------
    if train:
        # Color jitter
        jitter = torchvision.transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        )
        if random.random() < 0.8:
            img1 = jitter(img1)
        if random.random() < 0.8:
            img2 = jitter(img2)

        # # Gamma
        # if random.random() < 0.3:
        #     gamma = random.uniform(0.7, 1.5)
        #     img1 = img1 ** gamma
        #     img2 = img2 ** gamma
        #
        # # Gaussian noise
        # if random.random() < 0.2:
        #     noise_std = 0.02
        #     img1 = img1 + noise_std * torch.randn_like(img1)
        #     img2 = img2 + noise_std * torch.randn_like(img2)

    # -------------------------
    # Clamp & normalize images
    # -------------------------
    img1 = img1.clamp(0.0, 1.0)
    img2 = img2.clamp(0.0, 1.0)

    img1 = torchvision.transforms.functional.normalize(img1, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    img2 = torchvision.transforms.functional.normalize(img2, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    return img1, img2, flow, valid_flow_mask


def sintel_transform_simple(img1, img2, flow, valid_flow_mask=None, **kwargs):
    # --- Convert images to tensors + normalize ---
    img1 = torchvision.transforms.functional.to_tensor(img1)  # now in [0,1]
    img2 = torchvision.transforms.functional.to_tensor(img2)



    # Optional: Normalize images
    img1 = torchvision.transforms.functional.normalize(img1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img2 = torchvision.transforms.functional.normalize(img2, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    B, H, W = img1.shape
    # --- Convert flow (numpy) to Tensor ---
    if flow is not None:
        flow = torch.from_numpy(flow).float()  # shape = [2, H, W]

    # If a valid_flow_mask is provided, convert it
    if valid_flow_mask is not None:
        valid_flow_mask = torch.from_numpy(valid_flow_mask).bool()
    else:
        valid_flow_mask = torch.ones((1, H, W), dtype=torch.bool)

    H_new, W_new = 368, 768
    H, W = img1.shape[1:]
    img1 = img1[:, (H - H_new) // 2:(H + H_new) // 2, (W - W_new) // 2:(W + W_new) // 2]
    img2 = img2[:, (H - H_new) // 2:(H + H_new) // 2, (W - W_new) // 2:(W + W_new) // 2]
    valid_flow_mask = valid_flow_mask[:, (H - H_new) // 2:(H + H_new) // 2, (W - W_new) // 2:(W + W_new) // 2]
    flow = flow[:, (H - H_new) // 2:(H + H_new) // 2, (W - W_new) // 2:(W + W_new) // 2]

    #flow /= math.sqrt(flow.shape[1]**2 ++ flow.shape[2]**2)
    return img1, img2, flow, valid_flow_mask