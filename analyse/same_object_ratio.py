import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
import lightning as L
from torchvision.utils import save_image
from tqdm import tqdm
import clip
import sys
sys.path.append("..")

from solo.data.custom.imagenet import ImgNetDataset_42
from solo.data.custom.places import Places

parser = argparse.ArgumentParser()

# General
parser.add_argument('--min_crop', default=0.08, type=float)
parser.add_argument('--max_crop', default=1, type=float)
parser.add_argument('--dataset', default="imagenet", type=str)
parser.add_argument('--gpus', default=1, type=int)
args = parser.parse_args()

# ----------------------------
# CONFIG
# ----------------------------
if args.dataset == "imagenet":
    PATH = "/home/aubret/ImageNet/h5"
    dataset = ImgNetDataset_42(PATH)
elif args.dataset == "places":
    PATH = "/work/modelrep/schaumloeffel/Places365/"
    dataset = Places(PATH)
BATCH_SIZE = 256
NUM_PAIRS = 3_000_000
LOG_EVERY = 10_000
NUM_WORKERS = 1

# ----------------------------
# FABRIC SETUP
# ----------------------------
fabric = L.Fabric(accelerator="cuda", devices=args.gpus, strategy="ddp")
fabric.launch()

device = fabric.device

# ----------------------------
# TRANSFORMS
# ----------------------------
resnet_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(args.min_crop, args.max_crop)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

clip_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(args.min_crop, args.max_crop)),
    transforms.ToTensor(),
    transforms.Normalize(

        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
])

#0.48, 0.68, places 0.08
#0.39, 0.58, places 0.08->0.4
#0.28, 0.43, places 0.08->0.2


#0.7921: 0.08, 0.877
#0.8722: 0.2, 0.9462
#0.9052: 0.3, 0.9697
#0.9497: 0.5, 0.9891
#0.9615 :0.7, 0.9967
#0.9825 :0.9

# ----------------------------
# DATASET
# ----------------------------
class DualTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.base = dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return resnet_transform(img), resnet_transform(img) #, clip_transform(img)


dataset = DualTransformDataset(dataset)

sampler = DistributedSampler(
    dataset,
    num_replicas=fabric.world_size,
    rank=fabric.global_rank,
    shuffle=True
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)

loader = fabric.setup_dataloaders(loader)

# ----------------------------
# MODELS
# ----------------------------
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet.eval()

# clip_model, _ = clip.load("ViT-B/32", device=device)
# clip_model.eval()

resnet = fabric.setup(resnet)
# clip_model = fabric.setup(clip_model)

# CLIP text features
# classnames = [c[0] for c in datasets.ImageFolder(IMAGENET_PATH).classes]
# text_tokens = clip.tokenize(classnames).to(device)

# with torch.no_grad():
#     text_features = clip_model.encode_text(text_tokens)
#     text_features = F.normalize(text_features, dim=-1)

# ----------------------------
# MAIN LOOP
# ----------------------------
total_pairs = torch.tensor(0, device=device, dtype=torch.long)
same_resnet = torch.tensor(0, device=device, dtype=torch.long)
same_match = torch.tensor(0, device=device, dtype=torch.float)
same_clip = torch.tensor(0, device=device, dtype=torch.long)

pbar = tqdm(total=NUM_PAIRS, disable=fabric.global_rank != 0)

loader_iter = iter(loader)


def unnormalize(img, mean, std):
    """
    img: Tensor (C, H, W) normalized
    mean, std: lists of length C
    """
    mean = torch.tensor(mean).view(-1, 1, 1).to(img.device)
    std = torch.tensor(std).view(-1, 1, 1).to(img.device)
    return img * std + mean

total = 100000*fabric.global_rank
while total_pairs.item() < NUM_PAIRS:

    try:
        # x_resnet, x_clip = next(loader_iter)
        x_resnet, x_resnet2 = next(loader_iter)
    except StopIteration:
        sampler.set_epoch(torch.randint(0, 10_000, (1,)).item())
        loader_iter = iter(loader)
        # x_resnet, _x_clip = next(loader_iter)
        x_resnet, x_resnet2 = next(loader_iter)

    x_resnet = x_resnet.to(device, non_blocking=True)
    x_resnet2 = x_resnet2.to(device, non_blocking=True)
    # x_clip = x_clip.to(device, non_blocking=True)

    with torch.no_grad():
        # ResNet
        logits = resnet(x_resnet)
        pred_resnet = logits.argmax(dim=1)

        logits2 = resnet(x_resnet2)
        pred_resnet2 = logits2.argmax(dim=1)
        #
        # CLIP
        # img_feat = clip_model.encode_image(x_clip)
        # img_feat = F.normalize(img_feat, dim=-1)
        # logits_clip = img_feat @ text_features.T
        # pred_clip = logits_clip.argmax(dim=1)

    bs = x_resnet.size(0)

    # sample random pairs inside batch
    idx1 = torch.randint(0, bs, (bs,), device=device)
    idx2 = torch.randint(0, bs, (bs,), device=device)

    r1, r2 = pred_resnet, pred_resnet2
    # c1, c2 = pred_clip[idx1], pred_clip[idx2]

    same_resnet += (r1 == r2).sum()


    def top5_cross_agreement(logits1, logits2):
        """
        logits1, logits2: (B, num_classes)
        returns: (B,) bool tensor
        """
        pred1 = logits1.argmax(dim=1, keepdim=True)  # (B, 1)
        top5_2 = logits2.topk(5, dim=1).indices  # (B, 5)

        match = (top5_2 == pred1).any(dim=1)  # (B,)
        return match

    def symmetric_top5_agreement(logits1, logits2):
        m1 = top5_cross_agreement(logits1, logits2)
        m2 = top5_cross_agreement(logits2, logits1)
        return m1 & m2  # stricter


    matches = symmetric_top5_agreement(logits, logits2)
    # ratio = matches.float().mean()
    same_match += matches.float().sum()



    # idx = ~matches
    # if torch.any(idx) and fabric.global_rank == 0 and total < 100000*fabric.global_rank+200:
    #     os.makedirs("/scratch_shared/aubret/different_images3/", exist_ok=True)
    #     idx1, idx2 = x_resnet[idx], x_resnet2[idx]
    #     for i1, i2 in zip(idx1, idx2):
    #         # print(i, idx, flush=True)
    #         # print(x_resnet.shape, flush=True)
    #         mean = [0.485, 0.456, 0.406]
    #         std = [0.229, 0.224, 0.225]
    #         img_unnorm = unnormalize(i1, mean, std).clamp(0, 1)
    #         img_unnorm2 = unnormalize(i2, mean, std).clamp(0, 1)
    #         save_image(img_unnorm, f"/scratch_shared/aubret/different_images3/{total}_1.png")
    #         save_image(img_unnorm2, f"/scratch_shared/aubret/different_images3/{total}_2.png")
    #         total+=1


    # same_clip += (c1 == c2).sum()
    total_pairs += bs

    # ---- sync across GPUs ----

    total_pairs_all = fabric.all_reduce(total_pairs.clone(), reduce_op="sum")
    same_resnet_all = fabric.all_reduce(same_resnet.clone(), reduce_op="sum")
    global_matches = fabric.all_reduce(same_match.clone(), reduce_op="sum")

    if fabric.global_rank == 0:
        pbar.update(bs * fabric.world_size)

        # same_clip_all = fabric.all_reduce(same_clip.clone(), reduce_op="sum")
        if total_pairs_all.item() % LOG_EVERY < bs * fabric.world_size:
            print(f"\n--- {total_pairs_all.item()} pairs ---")
            print(f"ResNet ratio: {(same_resnet_all / total_pairs_all).item():.4f}")
            print(f"ResNet ratio: {(global_matches/ total_pairs_all).item():.4f}")
            # print(f"CLIP   ratio: {(same_clip_all / total_pairs_all).item():.4f}")

pbar.close()

if fabric.global_rank == 0:
    print("\nFINAL RESULTS")
    total_pairs_all = fabric.all_reduce(total_pairs.clone(), reduce_op="sum")
    same_resnet_all = fabric.all_reduce(same_resnet.clone(), reduce_op="sum")
    global_matches = fabric.all_reduce(same_match, reduce_op="sum")

    # same_clip_all = fabric.all_reduce(same_clip.clone(), reduce_op="sum")
    print(f"ResNet ratio: {(same_resnet_all / total_pairs_all).item():.4f}")
    print(f"ResNet ratio: {(global_matches / total_pairs_all).item():.4f}")

    # print(f"CLIP   ratio: {(same_clip_all / total_pairs_all).item():.4f}")