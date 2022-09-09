
import numpy as np


def calc_float_task_split(mean_tp_vect, var_tp_vect, comm_cp_vect, theta, gamma):
    a_p_vect = comm_cp_vect + gamma * (comm_cp_vect ** 2)
    b_p_vect = mean_tp_vect + 2 * gamma * comm_cp_vect*mean_tp_vect + gamma * var_tp_vect
    k_p_vect = (b_p_vect/(2 * gamma * (mean_tp_vect ** 2))) * \
               (-1 + np.sqrt(1 + (4 * gamma * (mean_tp_vect ** 2) * np.maximum(theta - a_p_vect, np.zeros(a_p_vect.shape)))
                             / (b_p_vect ** 2)))
    return k_p_vect


def find_task_split(mean_tp_vect, var_tp_vect, comm_cp_vect, init_theta, gamma, total_task_num):
    tmp_big_theta = init_theta
    tmp_small_theta = 0

    # find big theta that can be used for binary search if the given is not big enough
    tmp_k_p_vect_big = np.around(calc_float_task_split(mean_tp_vect, var_tp_vect, comm_cp_vect, tmp_big_theta, gamma))
    tmp_k_p_sum_big = np.sum(tmp_k_p_vect_big)
    while tmp_k_p_sum_big < total_task_num:
        tmp_small_theta = tmp_big_theta
        tmp_big_theta = 2 * tmp_big_theta
        tmp_k_p_vect_big = np.around(calc_float_task_split(mean_tp_vect, var_tp_vect, comm_cp_vect, tmp_big_theta, gamma))
        tmp_k_p_sum_big = np.sum(tmp_k_p_vect_big)

    # binary search to find a right theta
    tmp_k_p_vect = np.around(calc_float_task_split(mean_tp_vect, var_tp_vect, comm_cp_vect,
                                                   (tmp_big_theta + tmp_small_theta) / 2, gamma))
    tmp_k_p_sum = np.sum(tmp_k_p_vect)
    while tmp_k_p_sum != total_task_num:
        # choose new big or small theta according to the task sum
        if tmp_k_p_sum > total_task_num:
            tmp_big_theta = (tmp_big_theta + tmp_small_theta) / 2
        elif tmp_k_p_sum < total_task_num:
            tmp_small_theta = (tmp_big_theta + tmp_small_theta) / 2

        # calculating new task sum
        tmp_k_p_vect = np.around(calc_float_task_split(mean_tp_vect, var_tp_vect, comm_cp_vect,
                                                       (tmp_big_theta + tmp_small_theta) / 2, gamma))
        tmp_k_p_sum = np.sum(tmp_k_p_vect)

    return tmp_k_p_vect


mean_vect_example = np.array([2, 1.3, 1.2], dtype='float')
var_vect_example = np.array([0.2, 0.2, 0.2], dtype='float')
comm_vect_example = np.array([0.5, 0.5, 0.5], dtype='float')
init_big_theta = 100
gamma = 1
tot_task_num = 128

task_vect = find_task_split(mean_vect_example, var_vect_example, comm_vect_example, init_big_theta, gamma, tot_task_num)
print(task_vect)

