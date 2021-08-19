"""Auxiliary utils for implementing pruning strategies
"""

from collections import OrderedDict, defaultdict

import numpy
import torch
from torch import nn
from detectron2.structures.image_list import ImageList
from detectron2.structures.instances import Instances


def hook_applyfn(hook, model, forward=False, backward=False):
    """

    [description]

    Arguments:
        hook {[type]} -- [description]
        model {[type]} -- [description]

    Keyword Arguments:
        forward {bool} -- [description] (default: {False})
        backward {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    assert forward ^ backward, \
        "Either forward or backward must be True"
    hooks = []

    def register_hook(module):
        if (
            not isinstance(module, nn.Sequential)
            and
            not isinstance(module, nn.ModuleList)
            and
            not isinstance(module, nn.ModuleDict)
            and
            not (module == model)
        ):
            if forward:
                hooks.append(module.register_forward_hook(hook))
            if backward:
                hooks.append(module.register_backward_hook(hook))

    return register_hook, hooks


def get_params(model, recurse=False):
    """Returns dictionary of paramters

    Arguments:
        model {torch.nn.Module} -- Network to extract the parameters from

    Keyword Arguments:
        recurse {bool} -- Whether to recurse through children modules

    Returns:
        Dict(str:numpy.ndarray) -- Dictionary of named parameters their
                                   associated parameter arrays
    """
    params = {k: v.detach().cpu().numpy().copy()
              for k, v in model.named_parameters(recurse=recurse)}
    return params


def get_activations(model, input):

    activations = OrderedDict()

    def store_activations(module, input, output):
        if isinstance(module, nn.ReLU):
            # TODO ResNet18 implementation reuses a
            # single ReLU layer?
            return
        assert module not in activations, \
            f"{module} already in activations"
        # TODO: Deal with the segmentation models correctly
        # if isinstance(output, torch.Tensor):
        #     output_data = output
        # elif isinstance(output, dict):
        #     output_data = list(output.values())[0]
        # elif isinstance(output, list):
        #     output_data = output[0].tensor
        # elif isinstance(output, tuple):
        #     output_data = output[0][0]
        # else:
        #     print(output)
        #     print(type(output))
        #     for t in output:
        #         try:
        #             print(t.shape)
        #         except:
        #             pass
        #         print('t', type(t))
        #         for tt in t:
        #             print('tt', type(tt))
        #             try:
        #                 print(tt.shape)
        #             except:
        #                 pass
        #             for ttt in tt:
        #                 print('ttt', type(ttt))
        #                 try:
        #                     print(ttt.shape)
        #                 except:
        #                     pass
        #     raise ValueError()
        #
        # if isinstance(input[0], torch.Tensor):
        #     input_data = input[0]
        # elif isinstance(input[0], dict):
        #     input_data = input[0]['image']
        # elif isinstance(input[0], list) or isinstance(input[0], ImageList):
        #     input_data = input[0][0]
        # else:
        #     print(input[0])
        #     print(type(input[0]))
        #     for t in input[0]:
        #         try:
        #             print(t.shape)
        #         except:
        #             pass
        #         print('t', type(t))
        #         for tt in t:
        #             print('tt', type(tt))
        #             try:
        #                 print(tt.shape)
        #             except:
        #                 pass
        #             for ttt in tt:
        #                 print('ttt', type(ttt))
        #                 try:
        #                     print(ttt.shape)
        #                 except:
        #                     pass
        #                 break
        #             break
        #         break
        #
        #     raise ValueError()
        #
        # if isinstance(output_data, Instances):
        #     output_data = output_data.get('objectness_logits')
        #     output_data = output_data.detach().cpu().numpy().copy()
        # elif isinstance(output_data, torch.Tensor):
        #     output_data = output_data.detach().cpu().numpy().copy()
        # elif not isinstance(output_data, torch.Tensor):
        #     print(f"Input data type {type(input_data)} --- Output data type {type(output_data)}")
        #     print(output_data)
        #     try:
        #         print(output_data.get('objectness_logits'))
        #     except:
        #         pass
        # if isinstance(input_data, torch.Tensor):
        #     input_data = input_data.detach().cpu().numpy().copy()
        # elif not isinstance(input_data, torch.Tensor):
        #     # raise ValueError()
        #     print(f"Input data type {type(input_data)} --- Output data type {type(output_data)}")
        #     print(input_data)
        #     try:
        #         print(input_data)
        #     except:
        #         pass

        activations[module] = (input_data, output_data,)


    fn, hooks = hook_applyfn(store_activations, model, forward=True)
    model.apply(fn)
    with torch.no_grad():
        model(input)

    for h in hooks:
        h.remove()

    return activations


def get_gradients(model, inputs, outputs):
    # TODO implement using model.register_backward_hook()
    # So it is harder than it seems, the grad_input contains also the gradients
    # with respect to the weights and so far order seems to be (bias, input, weight)
    # which is confusing
    # Moreover, a lot of the time the output activation we are looking for is the
    # one after the ReLU and F.ReLU (or any functional call) will not be called by
    # the forward or backward hook
    # Discussion here
    # https://discuss.pytorch.org/t/how-to-register-hook-function-for-functional-form/25775
    # Best way seems to be monkey patching F.ReLU & other functional ops
    # That'll also help figuring out how to compute a module graph
    pass


def get_param_gradients(model, inputs, outputs, loss_func=None, by_module=True):

    gradients = OrderedDict()

    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()

    training = model.training
    model.train()
    pred = model(inputs)
    loss = loss_func(pred, outputs)
    loss.backward()

    if by_module:
        gradients = defaultdict(OrderedDict)
        for module in model.modules():
            assert module not in gradients
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad and param.grad is not None:
                    gradients[module][name] = param.grad.detach().cpu().numpy().copy()

    else:
        gradients = OrderedDict()
        for name, param in model.named_parameters():
            assert name not in gradients
            if param.requires_grad and param.grad is not None:
                gradients[name] = param.grad.detach().cpu().numpy().copy()

    model.zero_grad()
    model.train(training)

    return gradients


def fraction_to_keep(compression, model, prunable_modules):
    """ Return fraction of params to keep to achieve desired compression ratio

    Compression = total / ( fraction * prunable + (total-prunable))
    Using algrebra fraction is equal to
    fraction = total/prunable * (1/compression - 1) + 1

    Arguments:
        compression {float} -- Desired overall compression
        model {torch.nn.Module} -- Full model for which to compute the fraction
        prunable_modules {List(torch.nn.Module)} -- Modules that can be pruned in the model.

    Returns:
        {float} -- Fraction of prunable parameters to keep to achieve desired compression
    """
    from src.metrics import model_size
    total_size, _ = model_size(model)
    prunable_size = sum([model_size(m)[0] for m in prunable_modules])
    nonprunable_size = total_size - prunable_size
    fraction = 1 / prunable_size * (total_size/compression - nonprunable_size)
    assert 0 < fraction <= 1, \
        f"Cannot compress to {1/compression} model with {nonprunable_size/total_size}" + \
        "fraction of unprunable parameters"
    return fraction
