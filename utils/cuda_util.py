""" Cuda util function

Author: Zhao Na, 2020
"""


def cast_cuda(input):
    if isinstance(input, list):
        for i in range(len(input)):
            input[i] = cast_cuda(input[i])
    else:
        return input.cuda()
    return input
