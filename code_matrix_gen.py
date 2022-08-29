
import numpy as np


def get_rnd_h(d, n):
    h_mat = np.random.rand(d-1, n)
    h_mat[:, n-1] = -np.sum(h_mat[:, 0:(n-1)], axis=1)
    return h_mat


def get_poss_b(h_mat):
    (d_tag, n) = h_mat.shape
    d = d_tag + 1
    b_mat = np.zeros([n, n])
    for i in range(n):
        nonzero_inx = np.mod(range(i, i+d), n)
        relevant_h = h_mat[:, nonzero_inx[1:]]
        relevant_v = -1 * h_mat[:, nonzero_inx[0]]
        relevant_s = np.linalg.solve(relevant_h, relevant_v)
        b_mat[i, nonzero_inx] = np.concatenate((np.ones(1), relevant_s), axis=0)
    return b_mat


def get_rnd_b_mat(d, n):
    h_mat = get_rnd_h(d, n)
    b_mat = get_poss_b(h_mat)
    return b_mat


def calc_agg_coeff(fin_job_b_mat):
    K, n = fin_job_b_mat.shape
    b_mat_t = np.transpose(fin_job_b_mat)
    b_mat_t_square = np.delete(b_mat_t, range(n-K), axis=0)
    ones_vect = np.ones((K, 1))
    a_coeff = np.linalg.solve(b_mat_t_square, ones_vect)
    return a_coeff


def get_code_coeff(b_mat):
    _, n = b_mat.shape
    d = np.count_nonzero(b_mat[0, :])
    k = n - d + 1
    omega = n/k
    c = d / n
    return k, omega, c


d1 = 4
n1 = 8

b_matrix = get_rnd_b_mat(d1, n1)
K1, Omega1, C1 = get_code_coeff(b_matrix)
print('Calculated B matrix = \n{}'.format(np.round(b_matrix, 3)))
print('K = {}, Omega = {}, C = {}'.format(K1, Omega1, C1))

finished_job_b_mat = np.delete(b_matrix, range(n1-K1), axis=0)
print('Finished Job B matrix = \n{}'.format(np.round(finished_job_b_mat, 3)))

a_vect = calc_agg_coeff(finished_job_b_mat)
print('vector of coefficients for aggregation = \n{}'.format(np.round(a_vect, 3)))

print('final aggregate vect = \n{}'.format(np.transpose(finished_job_b_mat)@a_vect))
