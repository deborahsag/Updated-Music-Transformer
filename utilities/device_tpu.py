# For all things related to devices
#### ONLY USE PROVIDED FUNCTIONS, DO NOT USE GLOBAL CONSTANTS ####

import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm


TORCH_CPU_DEVICE = torch.device("cpu")

if (os.environ['COLAB_TPU_ADDR']):
    print("----- Running on TPU -----")
    TORCH_TPU_DEVICE = xm.xla_device(devkind='TPU')
    global USE_TPU
    USE_TPU = True

elif(torch.cuda.device_count() > 0):
    print("----- Running on CUDA -----")
    TORCH_CUDA_DEVICE = torch.device("cuda")
else:
    print("----- WARNING: CUDA devices not detected. This will cause the model to run very slow! -----")
    print("")
    TORCH_CUDA_DEVICE = None

global USE_CUDA
if not USE_TPU:
    USE_CUDA = True
else:
    USE_CUDA = False

# use_cuda
def use_cuda(cuda_bool):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Sets whether to use CUDA (if available), or use the CPU (not recommended)
    ----------
    """

    global USE_CUDA
    USE_CUDA = cuda_bool

# get_device
def get_device():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Grabs the default device. Default device is CUDA if available and use_cuda is not False, CPU otherwise.
    ----------
    """

    if USE_TPU:
        return TORCH_TPU_DEVICE
    elif((not USE_CUDA) or (TORCH_CUDA_DEVICE is None)):
        return TORCH_CPU_DEVICE
    else:
        return TORCH_CUDA_DEVICE

# tpu_device
def tpu_device():
    return TORCH_TPU_DEVICE

# cuda_device
def cuda_device():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Grabs the cuda device (may be None if CUDA is not available)
    ----------
    """

    return TORCH_CUDA_DEVICE

# cpu_device
def cpu_device():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Grabs the cpu device
    ----------
    """

    return TORCH_CPU_DEVICE
