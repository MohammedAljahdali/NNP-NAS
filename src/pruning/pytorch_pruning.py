from torch.nn.utils import prune
import torch
import numpy as np
from src.strategies.utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    flatten_importances,
                    importance_masks,
                    activation_importance)

class FooBarPruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        importances = map_importances(np.abs, self.params())
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)

        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask


class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold, abs_val):
        self.threshold = threshold
        self.abs_val = abs_val

    def compute_mask(self, tensor, default_mask):
        if self.abs_val:
            return torch.abs(tensor) > self.threshold
        else:
            return tensor > self.threshold

def magnitude_pruning(model, params):
    for child in model.children():
        if isinstance(child, torch.nn.Linear):

            if params['percentage']:
                weights = (child.weight.cpu()).detach().numpy()
                threshold = np.percentile(np.abs(weights), int(params['threshold_1']))
            else:
                threshold = params['threshold_1']

            parameters_to_prune = [(child, "weight")]
            prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruning,
                                      threshold=threshold, abs_val=params['absolute'])

            prune.remove(child, "weight")
    return model