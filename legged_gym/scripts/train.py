import os

# Available dog names: a1, aliengo, anymal_c, b1, go1, go2, lite3, mini_cheetah, mini_point, solo, spot
DOG_NAMES = [ "go2"]

os.environ["DOG_NAMES"] = ",".join(DOG_NAMES)

# Server workarounds (must be set before importing torch/isaacgym):
# - PYTORCH_JIT=0: torch 1.10+cu113 NVRTC does not know sm_89 (RTX 4090)
# - CXX/CC/CPATH: conda g++ needed to JIT-build the gymtorch extension
os.environ.setdefault("PYTORCH_JIT", "0")
_prefix = os.environ.get("CONDA_PREFIX", "")
if _prefix:
    os.environ.setdefault("CXX", "x86_64-conda-linux-gnu-c++")
    os.environ.setdefault("CC", "x86_64-conda-linux-gnu-gcc")
    os.environ.setdefault("CPATH", os.path.join(_prefix, "include"))

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.train import *

CUDA_DEVICE_ID = 0  # physical GPU 1 via CUDA_VISIBLE_DEVICES=1

args = get_args()
args.task = 'random_dog_stage1'
args.num_envs = 4096
args.headless = True
cuda = f"cuda:{CUDA_DEVICE_ID}"
args.rl_device = cuda
args.render_device = cuda
args.sim_device = cuda
args.graphics_device_num = -1  # NVF graphics init segfaults on this server; training is headless anyway

args.seed = 42
args.algo = 'MGDP'

args.load_world_model_policy = False

args.output_name = os.path.join(LEGGED_GYM_ROOT_DIR, 'outputs/go2/MGDP/stage1/baseline2')

print("args.output_name:", args.output_name)
train(args)
