import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from dgl.config import args

# plt.interactive(False)
def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.fromarray(buf, mode='RGBA')


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer.buffer_rgba())
    return X


def gpu2cpu(tensor):
    return tensor.to('cpu').detach().numpy()


def show_graph_with_labels(adjacency_matrix, mylabels, pos=None):
    fig = plt.figure()
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, pos=pos, node_color=range(adjacency_matrix.shape[0]), cmap=plt.cm.hsv)
    img = fig2data(fig)
    fig.clear()
    plt.close(fig)
    return img

def sample2d(Nsample):
    hiddenv = np.meshgrid(np.linspace(-1, 1, Nsample), np.linspace(-1, 1, Nsample))
    v = np.concatenate((np.expand_dims(hiddenv[0].flatten(), 1),
                        np.expand_dims(hiddenv[1].flatten(), 1)), 1)
    return v


def create_grid(Nsample, width, pos, adj, epoch=0):
    count = 0
    result_figsize_resolution = 40
    fig, axarr = plt.subplots(Nsample, Nsample, figsize=(result_figsize_resolution, result_figsize_resolution))
    for i in range(Nsample):
        for j in range(Nsample):
            graph = show_graph_with_labels(gpu2cpu(adj[Nsample * i + j]), range(500), pos=gpu2cpu(pos[Nsample * i + j]))
            axarr[i, j].imshow(graph)
            # plt.imsave(args.visualization+str(i)+str(j)+'.png', graph)
            plt.axis('off')
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_aspect('equal')
            count += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(args.visualization+'image_epoch_'+str(epoch)+'.png')
    fig.clear()
    plt.close(fig)
    # plt.show()