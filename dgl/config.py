from argparse import ArgumentParser

parser = ArgumentParser(description='Read a graph, and produce a layout with tsNET(*).')

# Input
parser.add_argument('--dataset_folder', '-f', type=str, default='./data/', help='Save layout to the specified file.')
parser.add_argument('--dataset', '-d', type=str, default='can96_data', help='Save layout to the specified file.')

args = parser.parse_args()