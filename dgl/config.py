from argparse import ArgumentParser

parser = ArgumentParser(description='Read a graph, and produce a layout with tsNET(*).')

# Input
parser.add_argument('--dataset_folder', '-f', type=str, default='/home/creddy/Work/graph/dgl/data/', help='include backslash')
parser.add_argument('--dataset', '-d', type=str, default='can_96', help='dataset format stop have _data')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='batchsize')
parser.add_argument('--device', type=str, default='cuda', help='gpu/cpu')

args = parser.parse_args()