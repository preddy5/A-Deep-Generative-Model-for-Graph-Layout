import torch


def d2_distance_matrix(dpos):
    dt = dpos.T
    dt_rep = dt[:, :, None,:].repeat(96, 2)
    dt_rep_t = dt_rep.transpose([0, 2, 1, 3])
    sub_dt = dt_rep_t - dt_rep
    sq_sub_dt = sub_dt**2
    dist_matrix = torch.sqrt(sq_sub_dt[:, :, :, 0] + sq_sub_dt[:,:,:,1])
    return dist_matrix
