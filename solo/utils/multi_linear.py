import math
from collections import OrderedDict
from typing import List, Dict, Any, Union, Optional, Generator, Tuple
from itertools import product

import omegaconf.listconfig
import torch
import torch.nn as nn

from solo.utils.misc import BatchSplitConcat, BatchMinus, BatchCatMinus, BatchCatCorrelate, BatchCorrelate


def parameter_iterator(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of parameters from a dictionary of lists.
    """

    keys, values = [], []
    for k, v in params.items():
        if v is not None:
            keys.append(k)
            values.append(v)
    return [dict(zip(keys, combination)) for combination in product(*values)]


# class LinearClassifier(nn.Module):
#     """Linear layer to train on top of frozen features"""
#
#     def __init__(
#             self,
#             out_dim: int,
#             num_classes: int,
#             **output_kwargs,
#     ):
#         super().__init__()
#         self.out_dim = out_dim
#         self.num_classes = num_classes
#         self.output_kwargs = output_kwargs
#
#         self.linear = nn.Linear(out_dim, num_classes)
#         self.linear.weight.data.normal_(mean=0.0, std=0.01)
#         self.linear.bias.data.zero_()
#
#     def forward(self, x):
#         ret, output = create_linear_input_vit(x, **self.output_kwargs)
#
#         if not ret:
#             raise RuntimeError(f"Linear layer returned no output.")
#         if output.shape[-1] != self.out_dim:
#             raise ValueError(
#                 f"Output shape {output.shape[-1]} does not match the expected input dimension {self.out_dim} of clf {self.use_avgpool=}, {self.use_n_blocks=}"
#             )
#         return self.linear(output)

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(
            self,
            out_dim: int,
            num_classes: int,
            depth: int = 1
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        if depth == 4:
            self.linear = nn.Sequential(
                BatchSplitConcat(),
                nn.Linear(out_dim * 2, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes)
            )
        elif depth == 5:
            self.linear = nn.Sequential(
                BatchSplitConcat(),
                nn.Linear(out_dim * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )
        elif depth == 3:
            self.linear = nn.Sequential(
                BatchSplitConcat(),
                nn.Linear(out_dim * 2, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )
        elif depth == 2:
            self.linear = nn.Sequential(
                BatchSplitConcat(),
                nn.Linear(out_dim * 2, num_classes)
            )
            self.linear[1].weight.data.normal_(mean=0.0, std=0.01)
            self.linear[1].bias.data.zero_()
        else:
            self.linear = nn.Linear(out_dim, num_classes)
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            self.linear.bias.data.zero_()

    def forward(self, x):
        if x.shape[-1] != self.out_dim:
            raise ValueError(f"Output shape {x.shape[-1]} does not match the expected input dimension {self.out_dim}.")
        return self.linear(x)

class DepthLinearClassifier(nn.Module):
    def __init__(
            self,
            out_dim: int,
            num_classes,
            depth: int
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        img_size = 224

        # class Interpolate(nn.Module):
        #     def forward(self, x):
        #         return torch.nn.functional.interpolate(x, size=(img_size, img_size), mode='bilinear',
        #                                                align_corners=False)
        if depth == -1:
            # class Reshape(nn.Module):
            #     def forward(self, x):
            #         x = x[:,:,1:]
            #         return x.view(x.shape[0], x.shape[1], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])))
            self.head = nn.Sequential(
                # Reshape(),
                nn.Conv2d(out_dim, num_classes[0], 1, padding=0),
            )
        elif depth == 0:
            self.head = nn.Sequential(
                nn.Conv2d(out_dim, num_classes[0], 3, padding=1),
                nn.Softplus() if num_classes[0] == 1 else nn.Identity()
            )
        elif depth == 1:
            self.head = nn.Sequential(
                nn.Conv2d(out_dim, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes[0], 1),
                nn.Softplus() if num_classes[0] == 1 else nn.Identity()
            )
        elif depth == 3:
            self.head = nn.Sequential(
                nn.Conv2d(out_dim, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes[0], 1),
                nn.Softplus() if num_classes[0] == 1 else nn.Identity()
            )
        elif depth == 31:
            out_dim = out_dim * 3
            self.head = nn.Sequential(
                BatchCatMinus(),
                nn.Conv2d(out_dim, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes[0], 1),
                nn.Softplus() if num_classes[0] == 1 else nn.Identity()
            )
        elif depth == 5:
            out_dim = 2 * out_dim + 81
            self.head = nn.Sequential(
                BatchCatCorrelate(),
                nn.Conv2d(out_dim, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes[0], 1),
                nn.Softplus() if num_classes[0] == 1 else nn.Identity()
            )
        elif depth == 6:
            out_dim = 2 * out_dim + (2*8+1)**2
            self.head = nn.Sequential(
                BatchCatCorrelate(max_displacement=8),
                nn.Conv2d(out_dim, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes[0], 1),
                nn.Softplus() if num_classes[0] == 1 else nn.Identity()
            )
        elif depth == 7:
            out_dim = (2 * 8 + 1) ** 2
            self.head = nn.Sequential(
                BatchCorrelate(max_displacement=8),
                nn.Conv2d(out_dim, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes[0], 1),
                nn.Softplus() if num_classes[0] == 1 else nn.Identity()
            )
        elif depth == 8:
            out_dim = (2 * 8 + 1) ** 2
            self.head = nn.Sequential(
                BatchCorrelate(max_displacement=8),
                nn.Conv2d(out_dim, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes[0], 1),
                nn.Softplus() if num_classes[0] == 1 else nn.Identity()
            )
        elif depth == 9:
            self.head = nn.Sequential(
                nn.Conv2d(out_dim, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                BatchCorrelate(max_displacement=8),
                nn.Conv2d((2 * 8 + 1) ** 2, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes[0], 1),
                nn.Softplus() if num_classes[0] == 1 else nn.Identity()
            )
        elif depth == 10:
            self.head = nn.Sequential(
                nn.Conv2d(out_dim, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                BatchCorrelate(max_displacement=8),
                nn.Conv2d((2 * 8 + 1) ** 2,64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes[0], 1),
                nn.Softplus() if num_classes[0] == 1 else nn.Identity()
            )
        elif depth == 30:
            self.head = nn.Sequential(
                nn.ConvTranspose2d(out_dim, 256, 3, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, 3, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes[0], 1),
                nn.Softplus() if num_classes[0] == 1 else nn.Identity()
            )

    def forward(self, x):
        if x.shape[1] != self.out_dim:
            raise ValueError(
                f"Output shape {x.shape[1]} does not match the expected input dimension {self.out_dim}.")

        return self.head(x)


class ViTLinearClassifier(nn.Module):
    def __init__(
            self,
            out_dim: int,
            num_classes: int,
            depth: int,
            **output_kwargs,
    ):
        super().__init__()
        self.output_kwargs = output_kwargs
        if isinstance(num_classes, tuple) or isinstance(num_classes, omegaconf.listconfig.ListConfig):
            self.head = DepthLinearClassifier(out_dim, num_classes, depth)
            self.output_kwargs["use_map"] = True
            self.output_kwargs["use_avgpool"] = False
            self.output_kwargs["use_cls_token"] = False
        else:
            self.head = LinearClassifier(out_dim, num_classes, depth)


    def forward(self, x):
        ret, output = create_linear_input_vit(x, **self.output_kwargs)



        if not ret:
            raise RuntimeError(f"Linear layer returned no output.")
        return self.head(output)




class CNNLinearClassifier(nn.Module):
    def __init__(
            self,
            out_dim: int,
            num_classes,
            depth: int,
            **output_kwargs,
    ):
        super().__init__()
        self.output_kwargs = output_kwargs
        if  isinstance(num_classes, omegaconf.listconfig.ListConfig):
            self.head = DepthLinearClassifier(out_dim, num_classes, depth)
            self.pool = "none"
        else:
            self.head = LinearClassifier(out_dim, num_classes, depth)
            self.pool = "avg"

    def forward(self, x):
        ret, output = create_linear_input_cnn(x, **self.output_kwargs, pool=self.pool)
        if not ret:
            raise RuntimeError(f"Linear layer returned no output.")
        return self.head(output)


def create_linear_input_cnn(x: Dict[str, torch.Tensor], layer_name: str, pool: str = "avg") -> Tuple[bool, torch.Tensor]:
    """
    Create the input for the linear head from the output of the CNN.

    Args:
        x: The output of the CNN.
        layer_name: The name of the layer to use.
        pool: The pooling method to use. Can be "avg" or "flatten".
    """
    if layer_name not in x:
        print(f"Layer {layer_name} not found in output.")
        return False, None

    # use the specified layer
    intermediate_output = x[layer_name]  # (batch, dim, height, width)

    if len(intermediate_output.shape) > 3:
        if pool == "avg":
            output = torch.mean(intermediate_output, dim=(2, 3))
        elif pool == "flatten":
            output = intermediate_output.view(intermediate_output.shape[0], -1)
        elif pool == "none":
            output = intermediate_output
        else:
            raise ValueError(f"Pool {pool} is not supported.")
    else:
        output = intermediate_output
    return True, output.contiguous()


# class LinearClassifierSimple(nn.Module):
#     """Linear layer to train on top of frozen features"""
#
#     def __init__(self, out_dim: int, num_classes: int = 1000):
#         super().__init__()
#         self.out_dim = out_dim
#         self.num_classes = num_classes
#
#         self.linear = nn.Linear(out_dim, num_classes)
#         self.linear.weight.data.normal_(mean=0.0, std=0.01)
#         self.linear.bias.data.zero_()
#
#     def forward(self, x):
#         return self.linear(x)


class AllClassifiers(nn.Module):
    """
    A wrapper for multiple classifiers.
    """

    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)

class AllClassifiersAllOuts(nn.Module):
    """
    A wrapper for multiple classifiers.
    """

    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        out = OrderedDict()
        for k, v in self.classifiers_dict.items():
            o = inputs[k]
            if len(out) != len(inputs) -1:
                o = o.detach()
            out[k] = v.forward(o)
        return out

    def __len__(self):
        return len(self.classifiers_dict)

def scale_lr(learning_rates: float, batch_size: int, devices: int) -> float:
    return learning_rates * (batch_size * devices) / 256.0


def setup_linear_classifiers_only_lr(
        sample_output: torch.Tensor,
        learning_rates: List[float],
        batch_size: int,
        devices: int,
        num_classes: int,
        depth: int
) -> Union[AllClassifiers, List[Dict[str, Any]]]:
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for _lr in sorted(learning_rates):
        lr = scale_lr(_lr, batch_size, devices)
        out_dim = sample_output.shape[-1]

        linear_classifier = LinearClassifier(out_dim, num_classes=num_classes, depth=depth)
        clf_str = f"classifier-lr_{lr:.8f}".replace(".", ":")
        linear_classifiers_dict[clf_str] = linear_classifier

        optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)

    return linear_classifiers, optim_param_groups

def setup_linear_classifiers_multi_layer(
        sample_output: Dict[str, torch.Tensor],
        learning_rates: List[float],
        layer_names: List[str],
        batch_size: int,
        devices: int,
        num_classes: int,
        depth: int
) -> Union[AllClassifiers, List[Dict[str, Any]]]:
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []

    for _lr in learning_rates:
        lr = scale_lr(_lr, batch_size, devices)
        for layer_name in layer_names:
            out_dim = sample_output[layer_name].shape[1]

            linear_classifier = CNNLinearClassifier(out_dim, num_classes=num_classes, layer_name=layer_name, depth=depth)
            clf_str = f"classifier-layer={layer_name}-lr_{lr:.8f}".replace(".", ":")
            print(f"Adding {clf_str}")
            linear_classifiers_dict[clf_str] = linear_classifier

            optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)

    return linear_classifiers, optim_param_groups

def setup_linear_classifiers_transformer(
        batch_size: int,
        devices: int,
        num_classes: int,
        param_dict: Dict[str, List[Any]],
        sample_output: torch.Tensor,
        has_class_token: bool,
        depth: int

) -> Union[AllClassifiers, List[Dict[str, Any]]]:
    print("Setting up linear classifiers:")

    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []

    if not 'lr' in param_dict:
        raise ValueError("Learning rate must be specified in the parameter dictionary.")

    for param in parameter_iterator(param_dict):
        lr = param.pop('lr')
        lr = scale_lr(lr, batch_size, devices)

        if isinstance(num_classes, tuple) or isinstance(num_classes, omegaconf.listconfig.ListConfig):
            param["use_map"] = True
            param["use_avgpool"] = False
            param["use_cls_token"] = False
        else:
            param["use_map"] = False


        ret, out_dim = create_linear_input_vit(sample_output, has_class_token=has_class_token,  **param)
        if not ret:
            continue

        out_dim = out_dim.shape[1]
        linear_classifier = ViTLinearClassifier(out_dim, num_classes=num_classes, has_class_token=has_class_token, depth=depth,
                                                **param)

        clf_str = "classifier-" + "-".join([f'{key}={value}' for key, value in param.items()]) + f"-lr={lr:.8f}"
        print("\t-", clf_str)
        linear_classifiers_dict[clf_str.replace(".", ":")] = linear_classifier

        optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    return linear_classifiers, optim_param_groups


def setup_linear_classifiers(
        sample_output: torch.Tensor,
        learning_rates: List[float],
        has_class_token: bool,
        batch_size: int,
        devices: int,
        num_classes: int,
        is_transformer: bool,
        # CNN search params
        layer_names: List[str] = None,
        # transformer search params
        use_avgpool: List[bool] = None,
        use_cls_token: List[bool] = None,
        use_n_blocks: List[int] = None,
        depth = 1,
) -> Union[AllClassifiers, List[Dict[str, Any]]]:
    special_search = any([x is not None for x in [use_avgpool, use_cls_token, use_n_blocks, layer_names]])

    if is_transformer and special_search:
        return setup_linear_classifiers_transformer(
            sample_output=sample_output,
            has_class_token=has_class_token,
            batch_size=batch_size,
            devices=devices, num_classes=num_classes,
            depth=depth,
            param_dict=dict(use_avgpool=use_avgpool, use_cls_token=use_cls_token, use_n_blocks=use_n_blocks,
                            lr=learning_rates),
        )
    elif not is_transformer and special_search:
        return setup_linear_classifiers_multi_layer(
            sample_output=sample_output,
            learning_rates=learning_rates,
            layer_names=layer_names,
            batch_size=batch_size,
            devices=devices, num_classes=num_classes, depth=depth
        )
    else:
        return setup_linear_classifiers_only_lr(
            sample_output, learning_rates, batch_size, devices, num_classes, depth=depth
        )


def create_linear_input_vit(
        x_tokens: torch.Tensor,
        use_n_blocks: int,
        use_avgpool: bool,
        use_cls_token: bool,
        use_map: bool = False,
        has_class_token: bool = True,
) -> Tuple[bool, torch.Tensor]:
    """
    Create the input for the linear head from the output of the Vision Transformer.

    It could be either the concatenation of the class token of the last `use_n_blocks` blocks or
    the concatenation of the class token of the last `use_n_blocks` blocks and the average pooled
    version of the last block's patch tokens.

    Args:
        x_tokens: The output of the Vision Transformer. Shape (batch, layer, tokens, dim).
        use_n_blocks: The number of blocks to use.
        use_avgpool: Whether to use average pooling or not.
        has_class_token: Whether the transformer has a class token or not.
    """

    if not has_class_token and not use_avgpool:
        print("Cannot use `use_avgpool=False` when the transformer has no cls tokens.")
        return False, None
    if use_cls_token and not has_class_token:
        print("Cannot use `use_cls_token=True` when the transformer has no cls tokens.")
        return False, None
    if not use_cls_token and not use_avgpool and not use_map:
        print("Cannot use both `use_cls_token=False` and `use_avgpool=False` and `use_map=False`.")
        return False, None

    # use the last `use_n_blocks` blocks
    intermediate_output = x_tokens[:, -use_n_blocks:, :, :]  # (batch, n_last_blocks, tokens, dim)
    # print("---", use_av:gpool, use_cls_token, use_n_blocks, use_map)
    # print(x_tokens.shape, intermediate_output.shape)
    if has_class_token:
        if use_cls_token:
            # take the class token of the last `use_n_blocks` blocks
            cls = intermediate_output[:, :, 0, :].reshape(intermediate_output.shape[0],
                                                          -1)  # (batch, n_last_blocks * dim)

        if use_avgpool:
            # average pool the patch tokens of the last block
            avgpool = intermediate_output[:, :, 1, :].reshape(intermediate_output.shape[0],
                                                              -1)  # (batch, n_last_blocks * dim)

        if use_avgpool and use_cls_token:
            output = torch.cat((cls, avgpool), dim=-1)
            output = output.reshape(output.shape[0], -1)
        elif use_map:
            # print(intermediate_output.shape)
            output = torch.permute(intermediate_output[:, -1, 1:, :], (0, 2, 1))
            if output.shape[2] == 300:
                output = output.view(output.shape[0], -1, 15, 20)
            else:
                output = output.view(output.shape[0], -1, int(math.sqrt(output.shape[2])), int(math.sqrt(output.shape[2])))
        elif use_avgpool:
            output = avgpool
        elif use_cls_token:
            output = cls
        else:
            raise ValueError("Cannot have both `use_cls_token=False` and `use_avgpool=False`.")

    else:
        # take the average pool of the last `use_n_blocks` blocks and concatenate them
        output = torch.mean(intermediate_output, dim=2).reshape(intermediate_output.shape[0], -1)
    return True, output.contiguous()
