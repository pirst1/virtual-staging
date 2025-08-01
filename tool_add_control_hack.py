# Script to add pretrained inpainting ControlNet
import sys
import os
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CON_DIR = os.path.abspath(os.path.join(CUR_DIR, "..", "ControlNet"))
if CON_DIR not in sys.path:
    sys.path.insert(0, CON_DIR)

from cldm.model import create_model
from share import *
import torch

import torch
from packaging import version

def load_checkpoint(path, **kwargs):
  if version.parse(torch.__version__) >= version.parse("2.6.0"):
    return torch.load(path, weights_only=False, **kwargs)
  else:
    return torch.load(path, **kwargs)

# assert len(sys.argv) == 3, 'Args are wrong.'

# input_path = sys.argv[1]
# output_path = sys.argv[2]

input_path = "./models/v1-5-pruned.ckpt"
output_path = "./models/control_sd15_pretrained_inpainter.ckpt"

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)
                      ), 'Output path is not valid.'


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./models/control_v11p_sd15_inpaint.yaml')

pretrained_weights = load_checkpoint(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

print("Replacing init ControlNet parameters with pretrained inpainting ControlNet parameters.")
# Transfer pretrained inpaint ControlNet state to the full model.
pretrained_weights_control_inpainter = load_checkpoint(
    './models/control_v11p_sd15_inpaint.pth')
for k in pretrained_weights_control_inpainter.keys():
    target_dict[k] = pretrained_weights_control_inpainter[k].clone()
    print(
        f'These weights have been updated with pretrained control state: {k}')


model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')