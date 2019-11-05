import numpy as np
from copy import deepcopy

def leapfrog1(position, momentum, grad_U, epsilon):
    momentum -= epsilon * grad_U(position) / 2
    position += epsilon * momentum
    momentum -= epsilon * grad_U(position) / 2

    return position, momentum

def find_reasonable_epsilon(position_0, grad_U, U):
    """ Heuristic for choosing an initial value of epsilon """
    epsilon = 1.
    momentum_0 = np.random.normal(0., 1., len(position_0))
    pos_0 = deepcopy(position_0)
    mom_0 = deepcopy(momentum_0)
    # Figure out what direction we should be moving epsilon.
    position_1, momentum_1 = leapfrog1(pos_0, mom_0, grad_U, epsilon)

    current_U = U(position_0)
    proposed_U = U(position_1)
    current_K = np.sum(momentum_0 ** 2) / 2
    proposed_K = np.sum(momentum_1 ** 2) / 2

    logacceptprob = current_U - proposed_U + current_K - proposed_K
    a = 1. if logacceptprob > np.log(0.5) else -1.
    # Keep moving epsilon in that direction until acceptprob crosses 0.5.
    # while ( (acceptprob ** a) > (2. ** (-a))):
    while a * logacceptprob > -a * np.log(2):
        epsilon = epsilon * (2. ** a)
        pos_0 = deepcopy(position_0)
        mom_0 = deepcopy(momentum_0)
        position_1, momentum_1 = leapfrog1(pos_0, mom_0, grad_U, epsilon)

        proposed_K = np.sum(momentum_1 ** 2) / 2
        proposed_U = U(position_1)
        logacceptprob = current_U - proposed_U + current_K - proposed_K
    # print("find_reasonable_epsilon=", epsilon)

    return epsilon

def normalize_vec(vec):
    return vec/np.linalg.norm(vec)

def compute_acc_ratio(curr_pos, curr_mom, new_pos, new_mom, U):
    current_U = U(curr_pos)
    current_K = np.sum(curr_mom ** 2) / 2
    proposed_U = U(new_pos)
    proposed_K = np.sum(new_mom ** 2) / 2

    return np.exp(current_U - proposed_U + current_K - proposed_K)

def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    """ Compute the stop condition in the main loop
    dot(dtheta, rminus) >= 0 & dot(dtheta, rplus >= 0)

    INPUTS
    ------
    thetaminus, thetaplus: ndarray[float, ndim=1]
        under and above position
    rminus, rplus: ndarray[float, ndim=1]
        under and above momentum

    OUTPUTS
    -------
    criterion: bool
        return if the condition is valid
    """
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)
