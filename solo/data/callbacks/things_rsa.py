import os
from pathlib import Path
from typing import Any

import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, DistributedSampler
from lightning.pytorch.utilities.types import STEP_OUTPUT

from solo.data.custom.things import Things
from torchvision import transforms
from lightning.pytorch.callbacks import Callback
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

@torch.no_grad()
def extract_features(dataloader, pl_module, layers):
    all_embeddings = [[] for _ in range(len(layers)+1)]
    all_concepts = []
    max_layer = max(layers)
    for images, conceptid in dataloader:
        all_concepts.append(conceptid.to(pl_module.device).clone())
        out = pl_module.backbone(images.to(pl_module.device))
        store_id = 0
        all_embeddings[store_id].append(out.clone())
        store_id += 1

        if hasattr(pl_module, "projector"):
            for l in range(len(pl_module.projector)):
                if l > max_layer:
                    break
                out = pl_module.projector[l](out)
                if l in layers:
                    all_embeddings[store_id].append(out.clone())
                    store_id += 1
    return [torch.cat(e) for e in all_embeddings], torch.cat(all_concepts)


def rearrange(matrix, classes_src, classes_dest, to_remove_name=[]):
    removed = []
    if not to_remove_name:
        to_remove = (matrix.sum(1) == 0)
        for i in range(len(classes_src)):
            if to_remove[i]:
                classes_dest.remove(classes_src[i])
                removed.append(classes_src[i])
                # print("remove", classes_src[i])
    else:
        for r in to_remove_name:
            if r in classes_dest:
                classes_dest.remove(r)

    new_matrix = torch.zeros((len(classes_dest), len(classes_dest)), device=matrix.device)
    for i in range(new_matrix.shape[0]):
        for j in range(new_matrix.shape[1]):
            old_i = classes_src.index(classes_dest[i])
            old_j = classes_src.index(classes_dest[j])
            new_matrix[i,j] = matrix[old_i, old_j]

    return new_matrix, classes_dest, removed

def save_matrix(sim_matrix_final, labels, path, dir, name, batch_idx, epoch):


    plt.figure(figsize=(13, 10))
    mask = np.eye(sim_matrix_final.shape[0], dtype=bool)
    sns.heatmap(sim_matrix_final, mask=mask, xticklabels=labels, yticklabels=labels, cmap='viridis', annot=False)
    plt.title("Concept similarity matrix",fontsize=20)

    os.makedirs(os.path.join(path,dir, str(epoch)+"_"+str(batch_idx)), exist_ok=True)
    plt.savefig(os.path.join(path,dir,str(epoch)+"_"+str(batch_idx), f"{name}.pdf"))

class ThingsCallback(Callback):
    def __init__(self, cfg, aug_kwargs):
        """
        Args:
            eval_fn (callable): a function that takes (model, dataloader) and returns a float metric
            log_name (str): name under which to log the metric
        """
        super().__init__()
        self.log_name = "things"
        self.cfg=cfg
        self.ordered_classes = pd.read_csv(Path(self.cfg.path) / "ordered_classes.csv", header=None)[0].tolist()
        self.adj_ordered_classes = [c for c in self.ordered_classes if c not in ["elephant", "giraffe", "bottle"]]


    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        t = transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        dataset = Things(Path(self.cfg.path), transforms=t)
        self.dataloader = DataLoader(dataset, batch_size=64, pin_memory=True, sampler=DistributedSampler(dataset, shuffle=True, drop_last=False))
        self.unordered_classes = dataset.codes
        self.missing_ego4d_classes = ["elephant", "giraffe", "bottle"]
        self.ego4d_cooccurrences = torch.tensor(np.load(Path(self.cfg.path) / "co_matrix_ordered.npy"), device=pl_module.device, dtype=torch.float32)
        self.human_rdm = torch.tensor(np.load(Path(self.cfg.path) / "hum_rdm2.npy"), device=pl_module.device, dtype=torch.float32)
        self.cfg.perform_every_n_batches = int(trainer.estimated_stepping_batches * self.cfg.perform_every_n_batches / trainer.max_epochs)

        self.small_human_rdm, _, _ = rearrange(self.human_rdm, self.unordered_classes, self.ordered_classes,self.missing_ego4d_classes)

        self.tiny_classes = ["car","motorcycle","bus","refrigerator","bowl","banana","toilet","toothbrush"]
        self.tiny_ego4d_rdm, _, _ = rearrange(self.ego4d_cooccurrences, self.ordered_classes, self.tiny_classes,self.missing_ego4d_classes)

        # print(pl_module.backbone.)
    # def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     if trainer.current_epoch != 0:
    #         return
    #     pl_module.eval()
    #     correlations = self.eval_fc(self.dataloader, pl_module)
    #     if trainer.is_global_zero:
    #         pl_module.log_dict(correlations)
    #     pl_module.train()

    # def on_train_epoch_end(self, trainer, pl_module):
    #     if trainer.current_epoch % self.freq_epochs:
    #         return
    #     pl_module.eval()
    #     correlations = self.eval_fc(self.dataloader, pl_module)
    #     if trainer.is_global_zero:
    #         pl_module.log_dict(correlations)
    #     pl_module.train()


    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        if self.cfg.perform_every_n_batches is not None and batch_idx % self.cfg.perform_every_n_batches == 0:
            # print("Start", flush=True)

            pl_module.eval()
            # pl_module.backbone.broadcast_buffers = False
            correlations = self.eval_fc(trainer, self.dataloader, pl_module, batch_idx, trainer.current_epoch)
            pl_module.log_dict(correlations, sync_dist=True, on_step=True, on_epoch=False)

            # pl_module.backbone.broadcast_buffers = True
            pl_module.train()





    @torch.no_grad()
    def eval_fc(self, trainer, dataloader, pl_module, batch_idx, epoch):
        features, conceptids = extract_features(dataloader, pl_module, self.cfg.layers)

        correlates = {}
        for i, f in enumerate(features):
            f = f.squeeze()
            # trainer.strategy.barrier()
            features_all, conceptids_all = pl_module.all_gather((f, conceptids))
            features_all, conceptids_all = (features_all.flatten(0,1),
                                            conceptids_all.flatten(0,1))

            model_rdm, count_rdm = self.machine_rdm(f, conceptids, features_all, conceptids_all)

            #RSA with human judgements
            model_rdms, count_rdms = pl_module.all_gather((model_rdm, count_rdm))
            model_rdms, count_rdms = model_rdms.sum(0), count_rdms.sum(0)

            model_rdms = model_rdms / count_rdms
            name = "rsa/backbone" if i == 0 else f"rsa/projector{self.cfg.layers[i-1]}"
            correlates[name] = self.correlate_rdms(model_rdms, self.human_rdm)


            # print(f"after correlate1 rdms {i}", correlates[name], flush=True)

            #RSA with Ego4D co-occurrences
            reordered_rdm, labels, _ = rearrange(model_rdms, self.unordered_classes,self.ordered_classes,self.missing_ego4d_classes)
            # print(f"after rearrange rdms {i}", reordered_rdm.shape, reordered_rdm.device, flush=True)
            name_small_rsa = "small_rsa/backbone" if i == 0 else f"small_rsa/projector{self.cfg.layers[i-1]}"
            correlates[name_small_rsa] = self.correlate_rdms(reordered_rdm, self.small_human_rdm)

            name_small_rsa = "small_spearman/backbone" if i == 0 else f"small_spearman/projector{self.cfg.layers[i-1]}"
            correlates[name_small_rsa] = self.correlate_rdms(reordered_rdm, self.small_human_rdm, correlation="spearman")

            name_ego4d = "ego4d-co/backbone" if i == 0 else f"ego4d-co/projector{self.cfg.layers[i-1]}"
            correlates[name_ego4d] = self.correlate_rdms(self.ego4d_cooccurrences, reordered_rdm)

            name_ego4d = "ego4d-spearman/backbone" if i == 0 else f"ego4d-spearman/projector{self.cfg.layers[i-1]}"
            correlates[name_ego4d] = self.correlate_rdms(self.ego4d_cooccurrences, reordered_rdm, correlation="spearman")
            # print(f"after correlate2 rdms {i}", correlates[name_ego4d],flush=True)

            tiny_rdm, labels, _ = rearrange(reordered_rdm, self.ordered_classes,self.tiny_classes,self.missing_ego4d_classes)
            name_ego4d = "tiny-ego4d-spearman/backbone" if i == 0 else f"tiny-ego4d-spearman/projector{self.cfg.layers[i-1]}"
            correlates[name_ego4d] = self.correlate_rdms(self.tiny_ego4d_rdm, tiny_rdm, correlation="spearman")

            if trainer.global_rank == 0:
                name = "backbone" if i == 0 else f"projector{self.cfg.layers[i-1]}"
                save_matrix(reordered_rdm.cpu().numpy(), self.adj_ordered_classes, self.cfg.path_matrix, self.cfg.name, name, batch_idx, epoch)
            # torch.cuda.empty_cache()
        return correlates



    @torch.no_grad()
    def machine_rdm(self, features_local, concept_ids_local, features, concept_ids):
        # print("concept_ids min/max:", concept_ids_local.shape, concept_ids.min(), concept_ids.max(), flush=True)
        n = int(concept_ids.max().item()) + 1
        rdm = torch.zeros((n, n), device=features.device, dtype=torch.float32)
        counts = torch.zeros((n, n), device=features.device, dtype=torch.float32)

        batch_size = 32
        for f, cid in zip(features_local.split(batch_size), concept_ids_local.split(batch_size)):
            rand_ids = torch.randint(0, features.shape[0], (1000,), device=features_local.device)
            rand_feats = features[rand_ids]
            rand_cid = concept_ids[rand_ids]
            dis_matrix = torch.nn.functional.cosine_similarity(f.unsqueeze(1), rand_feats.unsqueeze(0), dim=2)
            dis_matrix = - dis_matrix

            a = cid.unsqueeze(1).expand(-1, rand_cid.size(0)).reshape(-1)
            b = rand_cid.unsqueeze(0).expand(cid.size(0), -1).reshape(-1)
            vals = dis_matrix.reshape(-1)

            rdm.index_put_((a, b), vals, accumulate=True)
            rdm.index_put_((b, a), vals, accumulate=True)
            counts.index_put_((a, b), torch.ones_like(vals), accumulate=True)
            counts.index_put_((b, a), torch.ones_like(vals), accumulate=True)

        return rdm, counts

    @torch.no_grad()
    def correlate_rdms(self,
                       rdm_1: torch.Tensor,
                       rdm_2: torch.Tensor,
                       correlation: str = "pearson",
                       ) -> torch.Tensor:
        assert rdm_1.shape[0] == rdm_1.shape[1], f"RDMs must be square {rdm_1.shape[1]} {rdm_1.shape[0]}"

        n = rdm_1.shape[0]
        if rdm_1.shape != rdm_2.shape:
            print(f"RDMs must have the same shape {rdm_1.shape} {rdm_2.shape}", flush=True)
            return torch.tensor(0.0, device=rdm_1.device)
        # Get upper triangle indices, excluding the diagonal
        triu_indices = torch.triu_indices(n, n, offset=1)

        # Extract upper triangular values
        rdm1_vals = rdm_1[triu_indices[0], triu_indices[1]]
        rdm2_vals = rdm_2[triu_indices[0], triu_indices[1]]
        # Remove NaNs
        valid_mask = ~(torch.isnan(rdm1_vals) | torch.isnan(rdm2_vals))
        rdm1_vals = rdm1_vals[valid_mask]
        rdm2_vals = rdm2_vals[valid_mask]


        if rdm1_vals.numel() == 0 or rdm2_vals.numel() == 0:
            # Return 0 or NaN safely to all ranks
            print("No value !", rdm1_vals.device, rdm1_vals.shape, rdm2_vals.device, rdm2_vals.shape, flush=True)
            return torch.tensor(0.0, device=rdm_1.device)

        if correlation.lower() == "pearson":
            rdm1_vals = rdm1_vals - rdm1_vals.mean()
            rdm2_vals = rdm2_vals - rdm2_vals.mean()
            rho = torch.sum(rdm1_vals * rdm2_vals) / (
                    torch.sqrt(torch.sum(rdm1_vals ** 2)) * torch.sqrt(torch.sum(rdm2_vals ** 2)) + 1e-8
            )
        elif correlation.lower() == "spearman":
            rdm1_vals = rdm1_vals.argsort().argsort().float()
            rdm2_vals = rdm2_vals.argsort().argsort().float()
            rdm1_vals = rdm1_vals - rdm1_vals.mean()
            rdm2_vals = rdm2_vals - rdm2_vals.mean()
            rho = torch.sum(rdm1_vals * rdm2_vals) / (
                    torch.sqrt(torch.sum(rdm1_vals ** 2)) * torch.sqrt(torch.sum(rdm2_vals ** 2)) + 1e-8
            )
        else:
            raise ValueError(f"Unsupported correlation type: {correlation}")

        return rho




