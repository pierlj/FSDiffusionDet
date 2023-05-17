"""
"""
import numpy as np
import math



def normalize(Y_in):
    maxcol = np.max(Y_in, axis=1)
    Y_in = Y_in - maxcol[:, np.newaxis]
    
    Y_out = np.exp(Y_in)
    Y_out = Y_out / (np.sum(Y_out, axis=1)[:, None])

    return Y_out


def entropy_energy(Y, unary, kernel, bound_lambda, batch=False):
    tot_size = Y.shape[0]
    pairwise = kernel.dot(Y)
    if batch == False:
        temp = (unary * Y) + (-bound_lambda * pairwise * Y)
        E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()
    else:
        batch_size = 1024
        num_batch = int(math.ceil(1.0 * tot_size / batch_size))
        E = 0
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, tot_size)
            temp = (unary[start:end] * Y[start:end]) + (-bound_lambda * pairwise[start:end] * Y[start:end])
            E = E + (Y[start:end] * np.log(np.maximum(Y[start:end], 1e-20)) + temp).sum()

    return E




def bound_update(unary, kernel, bound_lambda, bound_iteration=20, batch=False):
    """
    """
    oldE = float('inf')
    Y = normalize(-unary)
    E_list = []
    for i in range(bound_iteration):
        additive = -unary
        mul_kernel = kernel.dot(Y)
        Y = -bound_lambda * mul_kernel
        additive = additive - Y
        Y = normalize(additive)
        E = entropy_energy(Y, unary, kernel, bound_lambda, batch)
        E_list.append(E)
        # print('entropy_energy is ' +repr(E) + ' at iteration ',i)
        if (i > 1 and (abs(E - oldE) <= 1e-6 * abs(oldE))):
            # print('Converged')
            break

        else:
            oldE = E.copy()

    return Y


def entropy_energy_mod(Y, unary, kernel, bound_lambda, ft_scores, bound_eta):
    tot_size = Y.shape[0]
    pairwise = kernel.dot(Y)

    temp = (unary * Y) + (-bound_lambda * pairwise * Y) - (bound_eta * ft_scores * Y)
    E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()

    return E

def bound_update_mod(unary, kernel, bound_lambda, ft_scores, bound_eta, bound_iteration=20, batch=False):
    """
    """
    # unary = unary * 0

    oldE = float('inf')
    # Y = normalize(-unary)
    Y = normalize(ft_scores)
    E_list = []
    for i in range(bound_iteration):
        additive = -unary
        mul_kernel = kernel.dot(Y)
        Y = -bound_lambda * mul_kernel - bound_eta * Y * ft_scores
        additive = additive - Y
        Y = normalize(additive)
        E = entropy_energy_mod(Y, unary, kernel, bound_lambda, ft_scores, bound_eta)
        E_list.append(E)
        # print('entropy_energy is ' +repr(E) + ' at iteration ',i)
        if (i > 1 and (abs(E - oldE) <= 1e-6 * abs(oldE))):
            # print('Converged')
            break

        else:
            oldE = E.copy()

    return Y


def bound_update_kmeans(support_embeddings, query_embeddings, kernel, bound_lambda, ft_scores, bound_eta, bound_iteration=20, batch=False):
    """
    """
    # unary = unary * 0
    n_ways = 3
    k_shot = 10

    support_embeddings = support_embeddings.cpu().numpy()
    query_embeddings = query_embeddings.cpu().numpy()
    cluster_means = support_embeddings.copy()
    
    substract = cluster_means[:, None, :] - query_embeddings
    distance = np.linalg.norm(substract, axis=-1, ord=2).T ** 2

    oldE = float('inf')
    Y = normalize(-distance)
    # Y = normalize(ft_scores)
    E_list = []
    for i in range(bound_iteration):
        substract = cluster_means[:, None, :] - query_embeddings
        distance = np.linalg.norm(substract, axis=-1, ord=2).T ** 2
        additive = - distance
        mul_kernel = kernel.dot(Y)
        Y = -bound_lambda * mul_kernel - bound_eta * Y * ft_scores
        additive = additive - Y

        weights = np.zeros_like(Y)
        clust_assign = np.argmax(Y, axis=-1)
        weights[np.arange(clust_assign.size), clust_assign] = 1
        weights = weights.T
        update_mean = (weights.sum(axis=-1) != 0)
        cluster_means[update_mean] = weights[update_mean]@query_embeddings / weights[update_mean].sum(axis=-1, keepdims=True)

        ft_scores_reshaped = ft_scores.reshape(-1, n_ways, k_shot).mean(axis=-1)
        ft_scores_max = ft_scores_reshaped.max(axis=-1)
        ft_labels = ft_scores_reshaped.argmax(axis=-1)
        threshold = 0.0
        ft_labels[ft_scores_max < threshold] = -1 
        
        for c in range(n_ways):
            query_per_label = query_embeddings[ft_labels == c]
            slc = slice(c * k_shot, (c+1) * k_shot)
            n_replace = (~update_mean)[slc].sum()
            if n_replace < query_per_label.shape[0]:
                cluster_means[slc][~update_mean[slc]] = query_per_label[np.random.choice(query_per_label.shape[0], n_replace, replace=False)]

        tau = 1
        Y = normalize(additive/tau)
        
        E = entropy_energy_mod(Y, distance, kernel, bound_lambda, ft_scores, bound_eta)
        E_list.append(E)
        # print('entropy_energy is ' +repr(E) + ' at iteration ',i)
        if (i > 1 and (abs(E - oldE) <= 1e-6 * abs(oldE))):
            # print('Converged')
            break

        else:
            oldE = E.copy()

    return Y, cluster_means