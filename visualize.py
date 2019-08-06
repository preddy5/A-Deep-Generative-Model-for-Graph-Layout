
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Read a graph, and produce a layout with tsNET(*).')

    # Input
    parser.add_argument('input_graph')
    args = parser.parse_args()

    import os
    import time
    import graph_tool.all as gt
    import modules.layout_io as layout_io
    import modules.graph_io as graph_io
    import modules.distance_matrix as distance_matrix

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

    gt.graph_draw(g)
