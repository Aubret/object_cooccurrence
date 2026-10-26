import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from torchvision.transforms import v2 as trv2, InterpolationMode
from lightning.pytorch.callbacks import Callback

from solo.utils.misc import concat_all_gather_no_grad


class FrankensteinDataset(Dataset):

    label_to_class = {0: "bear", 1: "bunny", 2: "cat", 3: "elephant", 4: "frog", 5: "lizard", 6: "tiger", 7: "turtle", 8:"wolf"}
    all_subsets = ["features", "frankenstein", "fragmented"]

    def __init__(self, root_dir, subset_name, transform=None):
        """
        Returns a torch dataset object for the images in a given subset_name.

        Args:
            subset_name (str): Choose "blurred", "boxedfeatures", "geons", "realistic", "silhouette", "all". "all" will return all the images from all the subsets.
            root_dir (str, optional): where subdirectories (subsets) for the data can be found. Defaults to "./individual_objects/".
            transform (torch.transform, optional): torch compose . Defaults to None.
            resize (int, optional): resizes the image while loading the .jpg file. Defaults to 300.
        """
        self.subset_prefix = {"features": "", "frankenstein": "f", "fragmented": "o"}
        assert subset_name in list(self.subset_prefix.keys()) + ["all"]
        self.class_to_label = {n: i for (i, n) in self.label_to_class.items()}

        self.root_dir = root_dir
        self.transform = transform

        self.labels, self.image_file_names, self.subset, self.name_to_id = [], [], [], {}
        if subset_name != "all":
            self.labels, self.image_file_names = self.return_images_labels(subset_name)
            self.subset = [subset_name] * len(self.labels)
        else:
            for subset in self.subset_prefix.keys():
                labels, image_file_names = self.return_images_labels(subset)
                # self.images.extend(images)
                self.labels.extend(labels)
                self.image_file_names.extend(image_file_names)
                self.subset.extend([subset] * len(labels))
        print("Dataset of ", subset_name)

    def return_images_labels(self, subset_name):
        labels, image_file_names = [], []
        for l, c in self.label_to_class.items():
            self.data_path = os.path.join(self.root_dir, c)
            prefix = self.subset_prefix[subset_name]+c
            files = [f for f in sorted(os.listdir(self.data_path)) if f.startswith(prefix)]
            self.name_to_id.update({subset_name+name: i+len(self.name_to_id) for i, name in enumerate(files)})
            labels += [l] * len(files)
            image_file_names += files


        return labels, image_file_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label =  self.labels[idx]
        img = Image.open(os.path.join(self.root_dir, self.label_to_class[label], self.image_file_names[idx])).convert("RGB")
        subset, name = self.subset[idx], self.image_file_names[idx]

        label = torch.tensor(label).float()

        if self.transform:
            img = self.transform(img)
        return img, label, self.name_to_id[subset+name]


@torch.no_grad()
def get_features(dataloader, model, device, linear=nn.Identity(), layer="avgpool"):
    def feature(f, layer):
        # k = list(f.keys())[0]
        return torch.flatten(f[layer], 1)
    features, labels, img_ids = [], [], []
    for r in dataloader:
        img, label, img_id = r[0].to(device), r[1].to(device), r[2].to(device)
        f = linear(model(img))
        if isinstance(f, dict):
            f = feature(f, layer)
        features.append(f)
        labels.append(label)
        img_ids.append(img_id)

    features, labels, img_ids = torch.cat(features, dim=0), torch.cat(labels, dim=0), torch.cat(img_ids, dim=0)

    features = concat_all_gather_no_grad(features)
    labels = concat_all_gather_no_grad(labels)
    img_ids = concat_all_gather_no_grad(img_ids)

    return features, labels, img_ids


class ConfArrangementCallback(Callback):
    def __init__(self, cfg, aug_kwargs):
        """
        Args:
            eval_fn (callable): a function that takes (model, dataloader) and returns a float metric
            log_name (str): name under which to log the metric
        """
        super().__init__()
        self.log_name = "frankenstein"
        self.freq_epochs = cfg.freq_epochs
        self.cfg = cfg
        self.aug_kwargs = aug_kwargs



    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        mean, std, image_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), self.aug_kwargs.resize
        preprocess = trv2.Compose([trv2.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),   trv2.CenterCrop(image_size), trv2.ToImage(),
                             trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])

        dataset_train = FrankensteinDataset(self.cfg.path, subset_name="features", transform=preprocess)
        dataset_test = FrankensteinDataset(self.cfg.path, subset_name="frankenstein", transform=preprocess)
        self.dataloader_features = DataLoader(dataset_train, batch_size=32, shuffle=False, pin_memory=True, sampler=DistributedSampler(dataset_train, shuffle=False))
        self.dataloader_frankenstein = DataLoader(dataset_test, batch_size=32, shuffle=False, pin_memory=True, sampler=DistributedSampler(dataset_test, shuffle=False))


    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.freq_epochs:
            return
        self.run(trainer,pl_module)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch != 0:
            return
        self.run(trainer,pl_module)


    def run(self, trainer, pl_module):
        pl_module.eval()
        features, labels, img_ids = get_features(self.dataloader_features, pl_module.backbone, pl_module.device,
                                                 layer=self.cfg.layer)
        features_test, labels_test, img_ids_test = get_features(self.dataloader_frankenstein, pl_module.backbone,
                                                                pl_module.device, layer=self.cfg.layer)

        raw_acc = self.eval_fc(features, features, labels, labels, img_ids, img_ids)
        raw_acc2 = self.eval_fc2(features, features, labels, labels, img_ids, img_ids)

        frank_acc = self.eval_fc(features, features_test, labels, labels_test, img_ids, img_ids_test)
        gap_acc = frank_acc - raw_acc
        pl_module.log_dict({"silhouette_acc": raw_acc, "silhouettes_acc_2ways": raw_acc2 , "frankenstein_acc": frank_acc, "conf_acc": gap_acc},
                           sync_dist=True, on_epoch=True)

        pl_module.train()

    @torch.no_grad()
    def eval_fc(self, features, features_test, labels, labels_test, img_ids, img_ids_test):
        success = 0
        all = 0
        lunique = labels.unique()

        for l in labels.unique():
            mask_label_t = (labels_test == l)
            mask_label = (labels == l)

            # mask_no_label = ~mask_label

            ids_label = img_ids[mask_label]

            lunique_neg = lunique[lunique != l]
            masksneg = [labels == ln for ln in lunique_neg]
            featureneg = [features[masksneg[ln]] for ln in range(len(masksneg))]
            for id in ids_label:
                mask_no_id_t = mask_label_t & (img_ids_test != id)
                mask_id = mask_label & (img_ids == id)

                positives = features_test[mask_no_id_t]
                # negatives = features[mask_no_label]

                main_f = features[mask_id].repeat(100, 1)
                p = positives[torch.randint(0, len(positives), (100,), device=features.device)]

                correct = torch.nn.functional.cosine_similarity(main_f, p, dim=1)
                bcorr = torch.ones((100,), device=features.device, dtype=torch.bool)
                for fneg in featureneg:
                    n = fneg[torch.randint(0, len(fneg), (100,), device=features.device)]
                    bcorr = bcorr & (correct > torch.nn.functional.cosine_similarity(p, n, dim=1))

                # all_success = ( (correct > wrong1) & (correct > wrong2) ).float().sum()
                all_success = bcorr.float().sum()
                success += all_success
                all += 100

        return success / all

    @torch.no_grad()
    def eval_fc2(self, features, features_test, labels, labels_test, img_ids, img_ids_test):
        success = 0
        fails = 0
        lunique = labels.unique()

        for l in lunique:
            mask_label_t = (labels_test == l)
            mask_label = (labels == l)

            mask_no_label = ~mask_label

            ids_label = img_ids[mask_label]
            for id in ids_label:
                mask_no_id_t = mask_label_t & (img_ids_test != id)
                mask_id = mask_label & (img_ids == id)

                positives = features_test[mask_no_id_t]
                negatives = features[mask_no_label]

                p = positives[torch.arange(0, len(positives), device=features.device).repeat(len(negatives)).view(-1)]
                n = negatives[torch.arange(0, len(negatives), device=features.device).view(-1, 1).repeat(
                    (1, len(positives))).view(-1)]
                main_f = features[mask_id].repeat((len(positives) * len(negatives), 1))

                correct = torch.nn.functional.cosine_similarity(main_f, p, dim=1)
                wrong2 = torch.nn.functional.cosine_similarity(n, p, dim=1)

                all_success = (correct > wrong2).float().sum()
                success += all_success
                fails += len(positives) * len(negatives) - all_success
        return success / (fails + success)








