# from https://github.com/chengcli/snapy/blob/cli/tmp_add_restart/examples/nc2pt.py

"""
Read variables in a NetCDF file and write them to jit saved torch tensors.
Usage: python nc2pt.py input.nc output.pt
"""

from netCDF4 import Dataset
import torch
from typing import Dict

def __save_tensors(tensor_map: dict[str, torch.Tensor], filename: str):
    class TensorModule(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            for name, tensor in tensors.items():
                self.register_buffer(name, tensor)

    module = TensorModule(tensor_map)
    scripted = torch.jit.script(module)  # Needed for LibTorch compatibility
    scripted.save(filename)

def save_nc_as_pt(input_fname, output_fname):
    nc = Dataset(input_fname, 'r')

    data = {}
    for varname in nc.variables:
        var = nc.variables[varname][:]
        if var.ndim == 4: # (time, x1, x2, x3) -> (time, x3, x2, x1)
            data[varname] = torch.tensor(var).permute(0, 3, 2, 1).squeeze()
        elif var.ndim == 3: # (x1, x2, x3) -> (x3, x2, x1)
            data[varname] = torch.tensor(var).permute(2, 1, 0).squeeze()
        else:
            data[varname] = torch.tensor(var).squeeze()

    __save_tensors(data, output_fname)

def get_nc_time(input_fname):
    nc = Dataset(input_fname, 'r')
    return float(nc.variables["time"][:])

def load_tensors(filename: str) -> Dict[str, torch.Tensor]:
    '''
    Adapted from https://github.dev/chengcli/snapy/blob/cli/tmp_add_restart/examples/shock.cpp
    '''
    tensor_map: Dict[str, torch.Tensor] = {}
    module: torch.jit.ScriptModule = torch.jit.load(filename)

    for p in module.named_buffers(recurse=False):
        tensor_map[p[0]] = p[1]
    for p in module.named_parameters(recurse=False):
        tensor_map[p[0]] = p[1]

    return tensor_map
