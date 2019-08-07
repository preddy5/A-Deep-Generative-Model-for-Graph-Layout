
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import graph_tool.all as gt
import modules.layout_io as layout_io
import modules.graph_io as graph_io
import modules.distance_matrix as distance_matrix
from modules.distance_matrix import d2_distance_matrix

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Read a graph, and produce a layout with tsNET(*).')

    # Input
    parser.add_argument('input_graph')
    args = parser.parse_args()


    # Check for valid input
    print(args)
    assert(os.path.isfile(args.input_graph))
    graph_name = os.path.splitext(os.path.basename(args.input_graph))[0]

    # Read input graph
    print('Reading graph: {0}...'.format(args.input_graph), end=' ', flush=True)
    g = graph_io.load_graph(args.input_graph)
    print('Done.')

    print('Input graph: {0}, (|V|, |E|) = ({1}, {2})'.format(graph_name, g.num_vertices(), g.num_edges()))

    # Compute the shortest-path distance matrix.
    print('Computing SPDM...'.format(graph_name), end=' ', flush=True)
    X = distance_matrix.get_distance_matrix(g, 'spdm', verbose=False)
    print('Done.')

    num_per = 12
    C_range = [0.1, 0.3]
    p_range = [1.5, 2.5]
    mu_range = [0.0, 0.5]
    mu_p_range = [0.8, 1.2]


    for c in np.arange(C_range[0], C_range[1], ((C_range[1]-C_range[0])/num_per)):
        parameters = []
        data = []
        for p in np.arange(p_range[0], p_range[1], ((p_range[1]-p_range[0])/num_per)):
            for mu in np.arange(mu_range[0], mu_range[1], ((mu_range[1]-mu_range[0])/num_per)):
                for mu_p in np.arange(mu_p_range[0], mu_p_range[1], ((mu_p_range[1]-mu_p_range[0])/num_per)):
                    # print(c, p, mu, mu_p)
                    pos = gt.sfdp_layout(g, C=c, p=p, mu=mu, mu_p=mu_p)
                    dpos = pos.get_2d_array([0,1])
                    parameters.append([c, p, mu, mu_p])
                    data.append(dpos)
        print('Done')
        parameters = np.array(parameters)
        data = np.array(data)
        np.save('data/can96_parameters_'+str(c)[:4], parameters)
        np.save('data/can96_data_' + str(c)[:4], data)


