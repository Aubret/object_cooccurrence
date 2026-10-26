
# Your Self-Supervised Projection Head Captures Object Co-Occurrence Statistics

Code repository for the paper "Your Self-Supervised Projection Head Captures Object Co-Occurrence Statistics", 
accepted for presentation at NeurIPS, Paris, 2026

## Code preparation

python -m pip install -r requirements.txt

## Dataset prepration

First download the Imagenet-1k, then:

```
./dataset_scripts/create_imgnetv2.py --data_root IMGNET_PATH --data_dest IMGNET_PATH
```

## Run training with controlled co-occurrence statistics

For all scripts: change: ++data.train_path=IMGNET_PATH/../ ++data.val_path=IMGNET_PATH/../ ++checkpoint.dir=CHECKPOINT_PATH
```
python3 main_pretrain.py --config-path scripts/pretrain/imagenet/ --config-name simclr.yaml ++data.dataset=imagenet_im_context
python3 main_pretrain.py --config-path scripts/pretrain/imagenet/ --config-name mocov3.yaml ++data.dataset=imagenet_im_context
python3 main_pretrain.py --config-path scripts/pretrain/imagenet/ --config-name dino_vit.yaml ++data.dataset=imagenet_im_context
python3 main_pretrain.py --config-path scripts/pretrain/imagenet/ --config-name byol.yaml ++data.dataset=imagenet_im_context
python3 main_pretrain.py --config-path scripts/pretrain/imagenet/ --config-name lejepa.yaml ++data.dataset=imagenet_im_context

```

To play with parameters:
- same context probability: `++data.dataset_kwargs.p=0.3`
- number of contexts: `++data.dataset_kwargs.n_context=300`

## Analytical code 

The analytical code will be uploaded soon.

## Citation
To cite our [paper](https://arxiv.org/pdf/2610.23915):
```bibtex
@inproceedings{aubret_your_2026,
	title = {Your {Self}-{Supervised} {Projection} {Head} {Captures} {Object} {Co}-{Occurrence} {Statistics}},
	url = {https://openreview.net/forum?id=xIeX3NaTxT#discussion},
	abstract = {Self-Supervised Learning (SSL) has achieved impressive success in learning semantic visual representations, yet the underlying principles driving this success remain underexplored. In this work, we hypothesize that common SSL pretext tasks implicitly model object co-occurrence statistics, a fundamental cue for visual learning. To test this, we curate three datasets of segmented objects from existing vision benchmarks using a state-of-the-art segmentation model. Through experiments across many SSL models, we reveal a hierarchical encoding of semantic information: while the visual backbone captures object categories, the projection head specializes in encoding co-occurrences between object categories. As a result, the projection head outperforms the backbone in aligning with human judgments of inter-category similarity. Furthermore, by controlling co-occurrence patterns during pre-training, we demonstrate that encoding object co-occurrences can significantly accelerate the emergence of category-level representations. Our findings uncover a previously hidden learning principle in SSL and suggest a path toward designing more effective pretext tasks by explicitly leveraging object co-occurrence structure.},
	language = {en},
	urldate = {2026-10-02},
	booktitle = {The {Fortieth} {Annual} {Conference} on {Neural} {Information} {Processing} {Systems}},
	author = {Aubret, Arthur and Triesch, Jochen and Teulière, Céline},
	month = sep,
	year = {2026},
}
```
