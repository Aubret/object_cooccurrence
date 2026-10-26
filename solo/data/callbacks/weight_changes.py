from collections import defaultdict
import torch
import lightning as L


class LayerWeightChangeCallback(L.Callback):
    def __init__(self, layers=["layer1","layer2","layer3","layer4","projector.0","projector.1","projector.2"]):
        self.prev_weights = {}
        self.layers = layers

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):

        layer_deltas = defaultdict(list)
        for name, param in pl_module.named_parameters():

            if not param.requires_grad:
                continue

            current = param.detach().cpu()

            if name in self.prev_weights:
                delta = (torch.norm(current - self.prev_weights[name]) / (torch.norm(self.prev_weights[name]) + 1e-12))
                # e.g. encoder.blocks.3.attn.qkv.weight
                # layer_name = ".".join(name.split(".")[:-1])
                for l in self.layers:
                    if l in name:
                        layer_deltas[l].append(delta.item())

            self.prev_weights[name] = current.clone()

        metrics = {
            f"weight_change/{layer}": sum(vals) / len(vals) for layer, vals in layer_deltas.items()
        }

        pl_module.log_dict(metrics, on_epoch=True)