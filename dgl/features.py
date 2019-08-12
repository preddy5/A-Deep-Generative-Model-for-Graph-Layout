import numpy as np
import torch
from torch.autograd import Variable


def normalize(pos):
    return pos / torch.max(pos)


def d2_distance_matrix(dpos, norm = True):
    # dt = torch.transpose(dpos, 1, 2)
    dt_rep = dpos[:, :, None, :].repeat(1, 1, 96, 1)
    dt_rep_t = dt_rep.transpose(1, 2)
    sub_dt = dt_rep_t - dt_rep
    sq_sub_dt = sub_dt ** 2
    dist_matrix = sq_sub_dt[:, :, :, 0] + sq_sub_dt[:, :, :, 1]
    if norm:
        dist_matrix = normalize(dist_matrix)
    return dist_matrix


def rand_uniform2d(batch_size):
    """ This function generates 2D samples from a uniform distribution in a 2-dimensional space
        Args:
            batch_size (int): number of batch samples
        Return:
            torch.Tensor: tensor of size (batch_size, 2)
    """
    z = 2 * (np.random.uniform(size=(batch_size, 2)) - 0.5)
    return torch.from_numpy(z).type(torch.FloatTensor)


def rand_projections(embedding_dim, num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.
        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples
        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w ** 2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)


def _sliced_wasserstein_distance(encoded_samples,
                                 distribution_samples,
                                 num_projections=50,
                                 p=2,
                                 device='gpu'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.
        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')
        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections).to(encoded_samples.device)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()
