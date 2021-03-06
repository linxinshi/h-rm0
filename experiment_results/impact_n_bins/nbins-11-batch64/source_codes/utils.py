import csv
def get_qrels(QRELS_DEV):
    qrels = {}
    with open(QRELS_DEV, mode='r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            qid = row[0]
            did = row[2]
            if qid not in qrels:
                qrels[qid] = []
            qrels[qid].append(did)
    return qrels



def kernal_mus(n_kernels):
    """
    get the mu for each gaussian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    #bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    bin_size = 1.0 / (n_kernels - 1)  # geodesic score range from [0, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    #print (l_mu)
    return l_mu


def kernel_sigmas(n_kernels):
    """
    get sigmas for each gaussian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    #bin_size = 2.0 / (n_kernels - 1)
    bin_size = 1.0 / (n_kernels - 1)  # geodesic score range from [0, 1]
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)

    return l_sigma