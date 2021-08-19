"""Module for computing FLOPs from specification, not from Torch objects
"""
import numpy as np


def dense_flops(in_neurons, out_neurons):
    """Compute the number of multiply-adds used by a Dense (Linear) layer"""
    return in_neurons * out_neurons


# def _conv_flops_compute(input,
#                         weight,
#                         bias=None,
#                         stride=1,
#                         padding=0,
#                         dilation=1,
#                         groups=1):
#     assert weight.shape[1] * groups == input.shape[1]
#
#     batch_size = input.shape[0]
#     in_channels = input.shape[1]
#     out_channels = weight.shape[0]
#     kernel_dims = list(weight.shape[-2:])
#     input_dims = list(input.shape[2:])
#
#     paddings = padding if type(padding) is tuple else (padding, padding)
#     strides = stride if type(stride) is tuple else (stride, stride)
#     dilations = dilation if type(dilation) is tuple else (dilation, dilation)
#
#     output_dims = [0, 0]
#     output_dims[0] = (input_dims[0] + 2 * paddings[0] -
#                       (dilations[0] * (kernel_dims[0] - 1) + 1)) // strides[0] + 1
#     output_dims[1] = (input_dims[1] + 2 * paddings[1] -
#                       (dilations[1] * (kernel_dims[1] - 1) + 1)) // strides[1] + 1
#
#     filters_per_channel = out_channels // groups
#     conv_per_position_flops = int(_prod(kernel_dims)) * in_channels * filters_per_channel
#     active_elements_count = batch_size * int(_prod(output_dims))
#     overall_conv_flops = conv_per_position_flops * active_elements_count
#
#     bias_flops = 0
#     if bias is not None:
#         bias_flops = out_channels * active_elements_count
#
#     overall_flops = overall_conv_flops + bias_flops
#
#     return int(overall_flops)


def conv2d_flops(in_channels, out_channels, input_shape, kernel_shape,
                 padding='same', strides=1, dilation=1):
    """Compute the number of multiply-adds used by a Conv2D layer
    Args:
        in_channels (int): The number of channels in the layer's input
        out_channels (int): The number of channels in the layer's output
        input_shape (int, int): The spatial shape of the rank-3 input tensor
        kernel_shape (int, int): The spatial shape of the rank-4 kernel
        padding ({'same', 'valid'}): The padding used by the convolution
        strides (int) or (int, int): The spatial stride of the convolution;
            two numbers may be specified if it's different for the x and y axes
        dilation (int): Must be 1 for now.
    Returns:
        int: The number of multiply-adds a direct convolution would require
        (i.e., no FFT, no Winograd, etc)
    >>> c_in, c_out = 10, 10
    >>> in_shape = (4, 5)
    >>> filt_shape = (3, 2)
    >>> # valid padding
    >>> ret = conv2d_flops(c_in, c_out, in_shape, filt_shape, padding='valid')
    >>> ret == int(c_in * c_out * np.prod(filt_shape) * (2 * 4))
    True
    >>> # same padding, no stride
    >>> ret = conv2d_flops(c_in, c_out, in_shape, filt_shape, padding='same')
    >>> ret == int(c_in * c_out * np.prod(filt_shape) * np.prod(in_shape))
    True
    >>> # valid padding, stride > 1
    >>> ret = conv2d_flops(c_in, c_out, in_shape, filt_shape, \
                       padding='valid', strides=(1, 2))
    >>> ret == int(c_in * c_out * np.prod(filt_shape) * (2 * 2))
    True
    >>> # same padding, stride > 1
    >>> ret = conv2d_flops(c_in, c_out, in_shape, filt_shape, \
                           padding='same', strides=2)
    >>> ret == int(c_in * c_out * np.prod(filt_shape) * (2 * 3))
    True
    """
    # validate + sanitize input
    assert in_channels > 0
    assert out_channels > 0
    assert len(input_shape) == 2
    assert len(kernel_shape) == 2
    # padding = padding.lower()
    padding = padding if type(padding) is tuple else (padding, padding)
    try:
        strides = tuple(strides)
    except TypeError:
        # if one number provided, make it a 2-tuple
        strides = (strides, strides)
    dilation = dilation if type(dilation) is tuple else (dilation, dilation)

    # compute output spatial shape
    # based on TF computations https://stackoverflow.com/a/37674568
    # if padding in ['same', 'zeros']:
    #     out_nrows = np.ceil(float(input_shape[0]) / strides[0])
    #     out_ncols = np.ceil(float(input_shape[1]) / strides[1])
    # else:  # padding == 'valid'
    #     out_nrows = np.ceil((input_shape[0] - kernel_shape[0] + 1) / strides[0])  # noqa
    #     out_ncols = np.ceil((input_shape[1] - kernel_shape[1] + 1) / strides[1])  # noqa
    # src: https://deepspeed.readthedocs.io/en/latest/_modules/deepspeed/profiling/flops_profiler/profiler.html
    output_dims = [0, 0]
    output_dims[0] = (input_shape[0] + 2 * padding[0] -
                      (dilation[0] * (kernel_shape[0] - 1) + 1)) // strides[0] + 1
    output_dims[1] = (input_shape[1] + 2 * padding[1] -
                      (dilation[1] * (kernel_shape[1] - 1) + 1)) // strides[1] + 1

    # output_shape = (int(out_nrows), int(out_ncols))

    # work to compute one output spatial position
    nflops = in_channels * out_channels * int(np.prod(kernel_shape))

    # total work = work per output position * number of output positions
    return nflops * int(np.prod(output_dims))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
