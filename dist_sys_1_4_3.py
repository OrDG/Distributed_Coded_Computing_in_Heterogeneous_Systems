#!/usr/bin/env python
"""
Parallel Hello World
"""

from mpi4py import MPI
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import code_matrix_gen as genb
import task_vect_calc
import pandas as pd
import socket


def print_to_console(some_string):
    sys.stdout.write(some_string)
    sys.stdout.flush()


def get_mpi_data_type(data_type):
    mpi_data_type = MPI.UNDEFINED
    if data_type == 'float':
        mpi_data_type = MPI.FLOAT
    elif data_type == 'i':
        mpi_data_type = MPI.INT
    return mpi_data_type


# ######################################### Master Functions #######################################

def send_data_from_master_to_all_workers_and_wait(data, data_type, tags=None):
    send_reqs = []
    for num_worker_rank in range(1, size):
        mpi_data_type = get_mpi_data_type(data_type)
        if tags is not None:
            send_reqs.append(comm.Isend([np.array(data, dtype=data_type), mpi_data_type], dest=num_worker_rank,
                                        tag=tags[num_worker_rank - 1]))
        else:
            send_reqs.append(comm.Isend([np.array(data, dtype=data_type), mpi_data_type], dest=num_worker_rank))
    MPI.Request.waitall(send_reqs)


def send_model_from_master_to_all_workers_and_wait(nn_model):
    send_reqs = []
    for nun_worker_rank in range(1, size):
        for inx, param_i in enumerate(nn_model.parameters()):
            send_reqs.append(comm.Isend([np.array(param_i.data.cpu(), dtype='float'), MPI.FLOAT], dest=nun_worker_rank,
                                        tag=batch_size+inx))
    MPI.Request.waitall(send_reqs)


def send_partitions_from_mater_to_all_workers_and_wait(partition):
    send_req = []
    for num_worker_rank in range(1, size):
        send_req.append(comm.Isend([partition[(num_worker_rank - 1):(num_worker_rank + 1)], MPI.INT],
                                   dest=num_worker_rank))
    MPI.Request.waitall(send_req)


def get_dataset_train_loader_cifar10():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set_CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=transform)
    train_loader_CIFAR10 = torch.utils.data.DataLoader(train_set_CIFAR10, batch_size=batch_size,
                                                       shuffle=True, drop_last=True)
    # We use drop_last=True to avoid the case where the data / batch_size != int

    return train_loader_CIFAR10


def get_buffers_for_workers(buffer_size, buffer_type):
    buffers = []
    for num_worker_rank in range(1, size):
        buffers.append(np.empty(buffer_size, dtype=buffer_type))
    return buffers


def get_model_buffers_for_workers(nn_model, copies_per_worker):
    buffers = []
    for num_worker_rank in range(1, size):
        for inx in range(copies_per_worker[num_worker_rank - 1]):
            buffers.append([np.empty(param.size(), dtype='float') for param in nn_model.parameters()])
    return buffers


def get_open_receive_requests_from_workers(buffers, buffers_type, tag):
    receive_requests = []
    buffers_mpi_type = get_mpi_data_type(buffers_type)
    for num_worker_rank in range(1, size):
        receive_requests.append(comm.Irecv([buffers[num_worker_rank-1], buffers_mpi_type], source=num_worker_rank,
                                           tag=tag))
    return receive_requests


def get_open_receive_requests_for_gradients_form_workers(nn_buffers, nn_model):
    receive_requests = []
    for num_worker_rank in range(1, size):
        for nn_index, nn_param in enumerate(nn_model.parameters()):
            receive_requests.append(comm.Irecv([nn_buffers[num_worker_rank - 1][nn_index], MPI.FLOAT],
                                               source=num_worker_rank, tag=nn_index))
    return receive_requests


def get_open_receive_requests_for_tasks_and_grads_from_workers(task_partitions, tasks_split, nn_model, job_vect_buffer,
                                                               grad_buffer, m):
    recv_grad_requests = []  # for getting job's gradients
    recv_job_vect_requests = []  # for getting job's vector
    for num_worker_rank in range(1, size):
        for inx in range(tasks_split[num_worker_rank - 1]):
            recv_job_vect_requests.append(
                comm.Irecv([job_vect_buffer[task_partitions[num_worker_rank - 1] + inx, :],
                            MPI.FLOAT], source=num_worker_rank, tag=inx))
            for index_p, _ in enumerate(nn_model.parameters()):
                recv_grad_requests.append(comm.Irecv(
                    [grad_buffer[task_partitions[num_worker_rank - 1] + inx][index_p], MPI.FLOAT],
                    source=num_worker_rank, tag=m * (index_p + 1) + inx))
    return recv_job_vect_requests, recv_grad_requests


def get_receive_times_for_gradients_by_notifications(receive_requests_gradients, receive_requests_notifications):
    receive_times = np.zeros((num_workers,), dtype='float')
    while not MPI.Request.Testall(receive_requests_notifications):
        index_worker, _ = MPI.Request.waitany(receive_requests_notifications)
        MPI.Request.waitall(receive_requests_gradients[index_worker*num_params:(index_worker+1)*num_workers])
        receive_times[index_worker] = time.time()
    return receive_times


def get_tp_kp_and_iter_times_and_update_finished_tasks_grads_and_vects(
        started_iter_time, recv_task_vect_req, recv_gradient_req, grad_buffer, task_buffer_matrix,
        finished_tasks_grad_list, finished_tasks_mat, num_tasks):

    finished_iter_time = started_iter_time  # only initiation
    finished_tp_kp_time = np.ones((num_workers,), dtype='float') * started_iter_time
    for num_finished_task in range(num_tasks):
        index_finished_task, _ = MPI.Request.waitany(recv_task_vect_req)  # returns the finished req index
        MPI.Request.waitall(recv_gradient_req[index_finished_task * num_params:(index_finished_task + 1) * num_params])

        # sys.stdout.write('step = {}, indx finished job = {}, time since iter start = {}\n'
        #                     .format(i, index_finished_job, time.time()-start_iter_time))
        # sys.stdout.flush()

        finished_tasks_grad_list[num_finished_task][:] = grad_buffer[index_finished_task][:]
        finished_tasks_mat[num_finished_task, :] = task_buffer_matrix[index_finished_task, :]

        finished_iter_time = time.time() if num_finished_task == k - 1 else finished_iter_time

        update_tp_kp_time_if_needed(task_partition, task_split_vect, index_finished_task,
                                    finished_tp_kp_time)
    return finished_iter_time, finished_tp_kp_time


def append_part1_results_to_metadata_dict(start_time, end_tp_time, time_buffer, dict_part1):
    if np.all(end_tp_time > 0):
        time_buffer = np.array(time_buffer, dtype='float')
        start_tp_calculation_time = time_buffer[:, 0]
        end_tp_calculation_time = time_buffer[:, 1]

        tp_calc_time_vect = end_tp_calculation_time - start_tp_calculation_time
        cp_time_vect = start_tp_calculation_time - start_time
        tp_and_cp_time_vect = end_tp_time - start_time
        tp_grad_time_vect = tp_and_cp_time_vect - cp_time_vect - tp_calc_time_vect

        dict_part1['tp_time_vect_list'].append(tp_calc_time_vect + tp_grad_time_vect)
        dict_part1['tp_calc_time_vect_list'].append(tp_calc_time_vect)
        dict_part1['tp_grad_time_vect_list'].append(tp_grad_time_vect)
        dict_part1['cp_time_vect_list'].append(cp_time_vect)


def get_timely_params_from_part_1(metadata_dict):
    tp_time = np.array(metadata_dict['tp_time_vect_list'])
    mean_tp = np.mean(tp_time, axis=0)
    var_tp = np.var(tp_time, axis=0)
    mean_cp = np.mean(np.array(metadata_dict['cp_time_vect_list']), axis=0)
    mean_tp_grad = np.mean(np.array(metadata_dict['tp_grad_time_vect_list']), axis=0)
    mean_tp_calc = np.mean(np.array(metadata_dict['tp_calc_time_vect_list']), axis=0)
    return mean_tp, var_tp, mean_cp, mean_tp_grad, mean_tp_calc, tp_time


def get_timely_params_from_part2(tp_kp_time_list, time_iter_list):
    tp_kp_time_array = np.array(tp_kp_time_list, dtype='float')
    mean_tp_kp_times = np.mean(tp_kp_time_array, axis=0)
    var_tp_kp_times = np.var(tp_kp_time_array, axis=0)
    second_moment_tp_kp_times = var_tp_kp_times + (mean_tp_kp_times ** 2)

    prob_tp_kp_vect = mean_tp_kp_times + gamma * second_moment_tp_kp_times
    mismatch_param = np.var(prob_tp_kp_vect)

    mean_time_iteration = np.mean(np.array(time_iter_list, dtype='float'))

    return mean_tp_kp_times, var_tp_kp_times, mismatch_param, mean_time_iteration


def update_tp_kp_time_if_needed(task_partitions, tasks_split, index_finished_task, finished_tp_kp_times):
    workers_rank = np.argmax(task_partitions - index_finished_task > 0)  # finds the worker rank because np.argmax returns the first True index
    task_num = index_finished_task - task_partitions[workers_rank - 1]  # finds the jobs index per worker
    if task_num == (tasks_split[workers_rank - 1] - 1):
        finished_tp_kp_times[workers_rank - 1] = time.time()


# ########################################### Workers Functions #############################################

def worker_wait_and_receive_data_from_master(data_shape, data_type, tag=None):
    buffer = np.empty(data_shape, dtype=data_type)
    mpi_data_type = get_mpi_data_type(data_type)
    if tag:
        receive_request = comm.Irecv([buffer, mpi_data_type], source=0, tag=tag)
    else:
        receive_request = comm.Irecv([buffer, mpi_data_type], source=0)
    receive_request.wait()
    return buffer


def worker_wait_and_receive_model_params_from_master(nn_model):
    receive_param_requests = []
    parameters_buffer = [np.empty(param.size(), dtype='float') for param in nn_model.parameters()]
    for inx, _ in enumerate(nn_model.parameters()):
        receive_param_requests.append(comm.Irecv([parameters_buffer[inx], MPI.FLOAT], source=0,
                                                 tag=batch_size + inx))

    MPI.Request.waitall(receive_param_requests)
    return parameters_buffer


def get_relevant_batch_indices_and_coefficients_from_code_matrix(num_epoch, num_step, tasks_code_matrix):
    # get a code vect from the code matrix
    task_index = num_epoch * num_runs_per_epoc + num_step
    num_tasks_in_matrix = tasks_code_matrix.shape[0]
    code_index = task_index % num_tasks_in_matrix
    code_vect = tasks_code_matrix[code_index, :]

    relevant_partition_ind = partition_vect[code_vect != 0]  # vector of relevant partition indices
    non_zero_code_coeffs = code_vect[code_vect != 0]  # vector of non-zero coeffs
    return relevant_partition_ind, non_zero_code_coeffs


def preform_a_calculation_and_update_total_gradient(total_grad_list, images_for_calc, labels_for_calc,
                                                    relevant_coefficient):
    images_per_calc = torch.from_numpy(images_for_calc).float().to(device)
    labels_per_calc = torch.from_numpy(labels_for_calc).long().to(device)

    outputs = model(images_per_calc)
    loss = criterion(outputs, labels_per_calc)
    optimizer.zero_grad()  # zero (flush) the gradients from the previous iteration
    loss.backward()  # calculate the gradients w.r.t the loss function (saved in tensor.grad field)

    # aggregating the gradients of the calc and mul by relevant coefficients
    for index_param, param in enumerate(model.parameters()):
        total_grad_list[index_param] += np.array(param.grad.cpu(), dtype='float') * relevant_coefficient


def worker_send_and_wait_total_gradients(total_gradient_list, m, is_part_2=False, task_inx=None):
    send_requests_total_gradients = []
    if is_part_2:
        for inx, total_grad in enumerate(total_gradient_list):
            send_requests_total_gradients.append(
                comm.Isend([total_grad, MPI.FLOAT], dest=0, tag=m * (inx + 1) + task_inx))
    else:
        send_requests_total_gradients = [comm.Isend([total_gradient_list[inx], MPI.FLOAT], dest=0, tag=inx) for
                                         inx in range(num_params)]
    MPI.Request.waitall(send_requests_total_gradients)


def worker_send_data_to_master_and_wait(data, data_type, tag=None):
    mpi_data_type = get_mpi_data_type(data_type)
    if tag:
        send_request = comm.Isend([np.array(data, dtype=data_type), mpi_data_type], dest=0, tag=tag)
    else:
        send_request = comm.Isend([np.array(data, dtype=data_type), mpi_data_type], dest=0)
    send_request.wait()


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    num_workers = size - 1

    # code parameters
    # m_vect = 2 ** np.arange(2, 5)
    m = 32
    d = 5
    k = m - d + 1
    init_big_theta = 10 ** 2
    gamma = 1

    # hyper-parameters:
    batch_size = 256
    learning_rate = 0.001
    size_dataset = 50000

    # statistical parameters
    num_total_runs_tp = 100
    num_total_runs_tp_kp = 100
    num_runs_per_epoc = int(np.floor(size_dataset/batch_size))
    num_epocs_tp = int(np.ceil(num_total_runs_tp/num_runs_per_epoc))
    num_epocs_tp_kp = int(np.ceil(num_total_runs_tp_kp / num_runs_per_epoc))

    # Device configuration
    device_count = torch.cuda.device_count()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get ip
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)

    print_to_console('rank = {}, device = {}, ip addr = {}\n'.format(rank, device, ip_addr))

    # create model, send it to device
    model = torchvision.models.resnet18(weights=None).to(device).float()
    num_params = len(list(model.parameters()))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_print_cycle = 2
    num_steps = 10

    if rank == 0:  # ######################################### Master ################################################
        # ########### part 1: tp estimation #################
        metadata_list = []

        # Get code matrix
        full_code_matrix = genb.get_rnd_b_mat(d, m)
        k, omega_max, c = genb.get_code_coeff(full_code_matrix)

        for num_tasks in range(k, m+1):
            code_matrix = full_code_matrix[:num_tasks, :]
            omega = num_tasks / k

            # measures dict
            metadata_dict_part1 = {'cp_time_vect_list': [], 'tp_time_vect_list': [], 'tp_calc_time_vect_list': [],
                                   'tp_grad_time_vect_list': []}

            print_to_console('Master: code matrix = \n{}\n k = {}, omega = {}, c = {}\n'.format(code_matrix,
                                                                                                k, omega, c))

            # sends code matrix to each worker
            send_data_from_master_to_all_workers_and_wait(code_matrix, 'float')

            # Get dataset
            train_loader_CIFAR10 = get_dataset_train_loader_cifar10()

            for epoch in range(num_epocs_tp):
                for i, (images, labels) in enumerate(train_loader_CIFAR10):
                    if i == num_steps:
                        break
                    # sending train-data shapes to each worker
                    start_cp_time = time.time()

                    send_data_from_master_to_all_workers_and_wait(labels.size(), 'i')
                    send_data_from_master_to_all_workers_and_wait(images.size(), 'i')

                    # send new parameters of the model to the workers
                    send_model_from_master_to_all_workers_and_wait(model)

                    # sending train-data to each worker
                    send_data_from_master_to_all_workers_and_wait(images, 'float',
                                                                  tags=np.arange(start=2, stop=2*size, step=2))
                    send_data_from_master_to_all_workers_and_wait(labels, 'i',
                                                                  tags=np.arange(start=3, stop=2*size+1, step=2))

                    # receiving grads for single task from each worker
                    # create relevant buffers for grads
                    time_buffer_tp = get_buffers_for_workers([2], 'float')
                    grad_buffer_tp = get_model_buffers_for_workers(model, np.ones([num_workers], dtype='i'))

                    # opening recv req for finished notification and gradients
                    recv_req_time_tp = get_open_receive_requests_from_workers(time_buffer_tp, 'float', 888888)
                    recv_req_grad_tp = get_open_receive_requests_for_gradients_form_workers(grad_buffer_tp, model)

                    # waiting for notification from each worker and their grad
                    end_tp_time_vect = get_receive_times_for_gradients_by_notifications(recv_req_grad_tp,
                                                                                        recv_req_time_tp)

                    append_part1_results_to_metadata_dict(start_cp_time, end_tp_time_vect, time_buffer_tp,
                                                          metadata_dict_part1)

            # ######################### part 2:optimal load split #########################

            mean_tp_vect, var_tp_vect, mean_cp_vect, mean_tp_grad_vect, mean_tp_calc_vect, tp_time_array = \
                get_timely_params_from_part_1(metadata_dict_part1)

            print_to_console('Master:\n mean Tp = {}, var Tp = {}, mean cp = {}, mean tp-calc = {}, mean tp-grad = {}\n'
                             .format(mean_tp_vect, var_tp_vect, mean_cp_vect, mean_tp_calc_vect, mean_tp_grad_vect))

            # get optimal load split
            task_split_vect = np.array(task_vect_calc.find_task_split(mean_tp_vect, var_tp_vect, mean_cp_vect,
                                                                      init_big_theta, gamma, num_tasks), dtype='i')

            # sending all workers flag about a second measurement in the case of a non-uniform task split
            is_split_uni = np.array_equal(task_split_vect, np.ones_like(task_split_vect, dtype='i') *
                                          int(num_tasks/num_workers))

            send_data_from_master_to_all_workers_and_wait(int(is_split_uni), 'i')

            for num_split_run in range(2):
                if num_split_run == 1:
                    if is_split_uni:
                        break
                    else:
                        task_split_vect = np.ones_like(task_split_vect, dtype='i') * int(num_tasks/num_workers)
                        num_left_tasks = num_tasks - num_workers * int(num_tasks/num_workers)
                        for j in range(num_left_tasks):
                            task_split_vect[-j] += 1

                task_partition = np.array([np.sum(task_split_vect[:j]) for j in range(num_workers + 1)], dtype='i')

                print_to_console('Master: task split = {}\n'.format(task_split_vect))

                # measure lists
                tp_kp_time_vect_list = []
                t_iter_list = []

                # send task partition indices to each worker (they already have the matrix)
                send_partitions_from_mater_to_all_workers_and_wait(task_partition)

                for epoch in range(num_epocs_tp_kp):
                    for i, (images, labels) in enumerate(train_loader_CIFAR10):
                        if i == num_steps:
                            break
                        start_iter_time = time.time()
                        # sending train-data shapes to each worker
                        send_data_from_master_to_all_workers_and_wait(labels.size(), 'i')
                        send_data_from_master_to_all_workers_and_wait(images.size(), 'i')

                        # send new parameters of the model to the workers
                        send_model_from_master_to_all_workers_and_wait(model)

                        # sending train-data to each worker
                        send_data_from_master_to_all_workers_and_wait(
                            images, 'float', tags=np.arange(start=2, stop=2 * size, step=2))
                        send_data_from_master_to_all_workers_and_wait(
                            labels, 'i', tags=np.arange(start=3, stop=2 * size + 1, step=2))

                        # getting calculated gradients and job vectors from each worker
                        # opening buffers for Irecv

                        job_vect_buffer_matrix = np.empty([num_tasks, m], dtype='float')
                        grad_buffer_list = get_model_buffers_for_workers(model, task_split_vect)

                        # opening Irecv requests for all possible jobs
                        recv_job_vect_req, recv_grad_req = get_open_receive_requests_for_tasks_and_grads_from_workers(
                            task_partition, task_split_vect, model, job_vect_buffer_matrix, grad_buffer_list, m)

                        # initializing finished jobs mat and gradients
                        finished_jobs_mat = np.empty([num_tasks, m], dtype='float')
                        finished_jobs_grad_list = get_model_buffers_for_workers(model, task_split_vect)

                        # waiting for only k jobs to be finished, and saving their gradients and vectors

                        finished_iter_time, finished_tp_kp_time = \
                            get_tp_kp_and_iter_times_and_update_finished_tasks_grads_and_vects(
                                start_iter_time, recv_job_vect_req, recv_grad_req, grad_buffer_list,
                                job_vect_buffer_matrix, finished_jobs_grad_list, finished_jobs_mat, num_tasks)

                        tp_kp_time_vect_list.append(finished_tp_kp_time-start_iter_time)
                        t_iter_list.append(finished_iter_time-start_iter_time)

                mean_tp_kp_vect, var_tp_kp_vect, mismatch, mean_time_iter = get_timely_params_from_part2(
                    tp_kp_time_vect_list, t_iter_list)

                print_to_console('Master:\n mean Tp_kp = {}, var Tp_kp = {}\n mismatch = {}, mean Titer = {}\n'
                                 .format(mean_tp_kp_vect, var_tp_kp_vect, mismatch, mean_time_iter))

                metadata_list.append(np.array([m, d, num_tasks, k, omega, c, mean_tp_vect, var_tp_vect, mean_cp_vect,
                                               task_split_vect, mean_tp_kp_vect, var_tp_kp_vect, mismatch,
                                               mean_time_iter, mean_tp_calc_vect, mean_tp_grad_vect,
                                               tp_time_array]))

        df_metadata = pd.DataFrame(metadata_list, columns=['m', 'd', 'num_tasks', 'k', 'omega', 'c', 'mean_tp_vect',
                                                           'var_tp_vect', 'mean_comm_time_vect', 'task_split_vect',
                                                           'mean_tp_kp_vect', 'var_tp_kp_vect', 'mismatch',
                                                           'mean_time_iter', 'mean_tp_calc_vect', 'mean_tp_grad_vect',
                                                           'tp_time_array'])
        df_metadata.to_excel('/home/ubuntu/cloud/metadata_fixed_5.xlsx')

    else:  # ####################################### Workers #######################################################
        model.train()
        # ######################### part 1: tp estimation #############################
        single_calc_batch_size = int(batch_size / m)
        partition_vect = np.arange(start=0, stop=batch_size, step=single_calc_batch_size, dtype='i')  # vector of start indices for each possible calcs
        for num_tasks in range(k, m+1):
            # receive code matrix
            code_matrix = worker_wait_and_receive_data_from_master([num_tasks, m], 'float')

            for epoch in range(num_epocs_tp):
                for i in range(num_runs_per_epoc):
                    if i == num_steps:
                        break

                    labels_shape = worker_wait_and_receive_data_from_master([1], 'i')
                    images_shape = worker_wait_and_receive_data_from_master([4], 'i')

                    # recv new parameters from master and update them in your model
                    param_buffer = worker_wait_and_receive_model_params_from_master(model)

                    # recv labels and images
                    images = worker_wait_and_receive_data_from_master(images_shape, 'float', tag=rank*2)
                    labels = worker_wait_and_receive_data_from_master(labels_shape, 'i', tag=rank*2+1)

                    # get a code vect from the code matrix
                    relevant_partition_indices, non_zero_code_coefficients = \
                        get_relevant_batch_indices_and_coefficients_from_code_matrix(epoch, i, code_matrix)

                    # preforming specific job calculations
                    start_tp_calc_time = time.time()
                    tot_grad_list = [np.zeros(param.size(), dtype='float') for param in
                                     model.parameters()]  # initializing total gradient list
                    for calc_num, index_calc_part in enumerate(relevant_partition_indices):

                        preform_a_calculation_and_update_total_gradient(
                            tot_grad_list, images[index_calc_part:index_calc_part + single_calc_batch_size, :, :, :],
                            labels[index_calc_part:index_calc_part + single_calc_batch_size],
                            non_zero_code_coefficients[calc_num])

                        # print worker's learning status
                        if calc_num == 0 and (i + 1) % num_print_cycle == 0:
                            print_to_console('Worker {}: Epoch [{}/{}], Step [{}/{}]\n'
                                             .format(rank, epoch + 1, num_epocs_tp, i + 1, num_runs_per_epoc))

                    end_tp_calc_time = time.time()

                    worker_send_and_wait_total_gradients(tot_grad_list, m)

                    worker_send_data_to_master_and_wait([start_tp_calc_time, end_tp_calc_time], 'float', tag=888888)

            # ############################# part 2:optimal load split #############################
            is_split_uni = worker_wait_and_receive_data_from_master([1], 'i')

            # get task partition indices
            for num_split_run in range(2):
                if num_split_run == 1 and is_split_uni[0] == 1:
                    break
                code_mat_partition = worker_wait_and_receive_data_from_master([2], 'i')

                # get relevant subset of task matrix
                jobs_mat = code_matrix[code_mat_partition[0]:code_mat_partition[1], :]

                for epoch in range(num_epocs_tp_kp):
                    for i in range(num_runs_per_epoc):
                        if i == num_steps:
                            break
                        # recv labels and images shapes
                        labels_shape = worker_wait_and_receive_data_from_master([1], 'i')
                        images_shape = worker_wait_and_receive_data_from_master([4], 'i')

                        # recv new parameters from master and update them in your model
                        param_buffer = worker_wait_and_receive_model_params_from_master(model)

                        # recv labels and images
                        images = worker_wait_and_receive_data_from_master(images_shape, 'float', tag=rank * 2)
                        labels = worker_wait_and_receive_data_from_master(labels_shape, 'i', tag=rank * 2 + 1)

                        # extracting num tasks
                        k_p = jobs_mat.shape[0]
                        for task_p_inx in range(k_p):

                            # extracting specific job properties
                            task_p_vect = jobs_mat[task_p_inx, :]  # vector of coeff for the job
                            relevant_partition_indices_per_task = partition_vect[task_p_vect != 0]  # vector of relevant partition indices
                            non_zero_task_coefficients = task_p_vect[task_p_vect != 0]  # vector of non-zero coeffs

                            # preforming specific job calculations
                            tot_grad_list = [np.zeros(param.size(), dtype='float') for param in
                                             model.parameters()]  # initializing total gradient list
                            for calc_num, index_calc_part in enumerate(relevant_partition_indices_per_task):

                                preform_a_calculation_and_update_total_gradient(
                                    tot_grad_list, images[index_calc_part:index_calc_part + single_calc_batch_size, :, :, :],
                                    labels[index_calc_part:index_calc_part + single_calc_batch_size],
                                    non_zero_task_coefficients[calc_num])

                                # print worker's learning status
                                if calc_num == 0 and task_p_inx == 0 and (i + 1) % num_print_cycle == 0:
                                    print_to_console('Worker {}: Epoch [{}/{}], Step [{}/{}]\n'
                                                     .format(rank, epoch + 1, num_epocs_tp_kp, i + 1,
                                                             num_runs_per_epoc))

                            # sending total gradient of a job and its vector

                            worker_send_and_wait_total_gradients(tot_grad_list, m, is_part_2=True,
                                                                 task_inx=task_p_inx)

                            worker_send_data_to_master_and_wait(task_p_vect, 'float', tag=task_p_inx)
