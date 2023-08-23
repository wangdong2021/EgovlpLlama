import torch
from IPython import embed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda', index=args.index)
tensor_size = 22 << 28
tensor = torch.zeros(tensor_size, device=device)
embed()
