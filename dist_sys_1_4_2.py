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


def send_data_master_to_all_workers_and_wait(data, data_type, tags=None):
    send_reqs = []
    for num_worker_rank in range(1, size):
        mpi_data_type = get_mpi_data_type(data_type)
        if tags is not None:
            send_reqs.append(comm.Isend([np.array(data, dtype=data_type), mpi_data_type], dest=num_worker_rank,
                                       tag=tags[num_worker_rank - 1]))
        else:
            send_reqs.append(comm.Isend([np.array(data, dtype=data_type), mpi_data_type], dest=num_worker_rank))
    MPI.Request.waitall(send_reqs)


def send_model_master_to_all_workers_and_wait(nn_model):
    send_reqs = []
    for nun_worker_rank in range(1, size):
        for inx, param_i in enumerate(nn_model.parameters()):
            send_reqs.append(comm.Isend([np.array(param_i.data.cpu(), dtype='float'), MPI.FLOAT], dest=nun_worker_rank,
                                        tag=batch_size+inx))
    MPI.Request.waitall(send_reqs)


def get_dataset_train_loader_cifar10():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set_CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=transform)
    train_loader_CIFAR10 = torch.utils.data.DataLoader(train_set_CIFAR10, batch_size=batch_size,
                                                       shuffle=True, drop_last=True)
    # We use drop_last=True to avoid the case where the data / batch_size != int

    return train_loader_CIFAR10


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    num_workers = size - 1

    # code parameters
    n_vect = 2 ** np.arange(2, 5)
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
    # model = nn.DataParallel(model)
    # model.to(device).float()
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

        for n in n_vect:
            for d in range(1, n):
                # Get code matrix
                code_matrix = genb.get_rnd_b_mat(d, n)
                k, omega, c = genb.get_code_coeff(code_matrix)

                # measure_lists
                cp_time_vect_list = []
                tp_time_vect_list = []
                tp_calc_time_vect_list = []
                tp_grad_time_vect_list = []

                print_to_console('Master: code matrix = \n{}\n k = {}, omega = {}, c = {}\n'.format(code_matrix,
                                                                                                    k, omega, c))

                # sends code matrix to each worker
                send_data_master_to_all_workers_and_wait(code_matrix, 'float')

                # Get dataset
                train_loader_CIFAR10 = get_dataset_train_loader_cifar10()

                for epoch in range(num_epocs_tp):
                    for i, (images, labels) in enumerate(train_loader_CIFAR10):
                        if i == num_steps:
                            break
                        # sending train-data shapes to each worker
                        start_cp_time = time.time()

                        send_data_master_to_all_workers_and_wait(labels.size(), 'i')
                        send_data_master_to_all_workers_and_wait(images.size(), 'i')

                        # send new parameters of the model to the workers
                        send_model_master_to_all_workers_and_wait(model)

                        # sending train-data to each worker
                        send_data_master_to_all_workers_and_wait(images, 'float', tags=np.arange(start=2, stop=2*size, 
                                                                                                 step=2))
                        send_data_master_to_all_workers_and_wait(labels, 'i', tags=np.arange(start=3, stop=2*size+1, 
                                                                                             step=2))

                        # saving communication time, we assume the diff between cp of workers is negligible
                        # end_cp_time = time.time()
                        # cp_time_vect = end_cp_time_vect - start_cp_time
                        #comm_time.append(end_cp_time - start_cp_time)

                        # receiving grads for single task from each worker
                        # create relevant buffers for grads
                        grad_buffer_tp = []
                        time_buffer_tp = []
                        for worker_rank in range(1, size):
                            grad_buffer_tp.append([np.empty(param.size(), dtype='float') for param in model.parameters()])
                            time_buffer_tp.append(np.empty([2], dtype='float'))

                        # opening recv req for finished notification and gradients
                        recv_req_time_tp = []
                        recv_req_grad_tp = []
                        for worker_rank in range(1, size):
                            recv_req_time_tp.append(comm.Irecv([time_buffer_tp[worker_rank-1], MPI.FLOAT],
                                                               source=worker_rank, tag=888888))
                            for index, param in enumerate(model.parameters()):
                                recv_req_grad_tp.append(comm.Irecv([grad_buffer_tp[worker_rank-1][index], MPI.FLOAT],
                                                                   source=worker_rank, tag = index))

                        # waiting for notification from each worker and their grad
                        end_tp_time_vect = np.zeros((num_workers,), dtype=float)
                        while not MPI.Request.Testall(recv_req_time_tp):
                            worker_index, _ = MPI.Request.waitany(recv_req_time_tp)
                            MPI.Request.waitall(recv_req_grad_tp[worker_index*num_params:(worker_index+1)*num_params])
                            end_tp_time_vect[worker_index] = time.time()
                        # MPI.Request.waitall(recv_req_grad_tp)

                        if np.all(end_tp_time_vect > 0):
                            time_buffer_tp = np.array(time_buffer_tp, dtype='float')
                            start_tp_calc_time = time_buffer_tp[:, 0]
                            end_tp_calc_time = time_buffer_tp[:, 1]

                            tp_calc_time_vect = end_tp_calc_time - start_tp_calc_time
                            cp_time_vect = start_tp_calc_time - start_cp_time
                            tp_cp_time_vect = end_tp_time_vect - start_cp_time
                            tp_grad_time_vect = tp_cp_time_vect - cp_time_vect - tp_calc_time_vect

                            tp_time_vect_list.append(tp_calc_time_vect+tp_grad_time_vect)
                            tp_calc_time_vect_list.append(tp_calc_time_vect)
                            tp_grad_time_vect_list.append(tp_grad_time_vect)
                            cp_time_vect_list.append(cp_time_vect)

                # ########### part 2:optimal load split #################

                # calc params for optimal load split
                tp_time_array = np.array(tp_time_vect_list)
                mean_tp_vect = np.mean(tp_time_array, axis=0)
                var_tp_vect = np.var(tp_time_array, axis=0)
                mean_cp_vect = np.mean(np.array(cp_time_vect_list), axis=0)
                mean_tp_grad_vect = np.mean(np.array(tp_grad_time_vect_list), axis=0)
                mean_tp_calc_vect = np.mean(np.array(tp_calc_time_vect_list), axis=0)

                print_to_console('Master:\n mean Tp = {}, var Tp = {}, mean cp = {}, mean tp-calc = {}, mean tp-grad = {}\n'
                                 .format(mean_tp_vect, var_tp_vect, mean_cp_vect, mean_tp_calc_vect, mean_tp_grad_vect))

                # get optimal load split
                task_split_vect = task_vect_calc.find_task_split(mean_tp_vect, var_tp_vect, mean_cp_vect,
                                                                 init_big_theta, gamma, n)
                task_split_vect = np.array(task_split_vect, dtype='i')
                is_split_uni = np.array_equal(task_split_vect, np.ones_like(task_split_vect, dtype='i') *
                                              int(n/num_workers))

                send_req_is_split_uni = []
                for worker_rank in range(1, size):
                    send_req_is_split_uni.append(comm.Isend([np.array(int(is_split_uni), dtype='i'), MPI.INT], dest=worker_rank))
                MPI.Request.waitall(send_req_is_split_uni)

                for num_split_run in range(2):
                    if num_split_run == 1:
                        if is_split_uni:
                            break
                        else:
                            task_split_vect = np.ones_like(task_split_vect, dtype='i') * int(n/num_workers)
                            num_left_tasks = n - num_workers * int(n/num_workers)
                            for j in range(num_left_tasks):
                                task_split_vect[-j] += 1

                    task_partition = np.array([np.sum(task_split_vect[:j]) for j in range(num_workers + 1)], dtype='i')

                    print_to_console('Master: task split = {}\n'.format(task_split_vect))

                    # measure lists
                    tp_kp_time_vect_list = []
                    t_iter_list = []

                    # send task partition indices to each worker (they already have the matrix)
                    send_req_partition = []
                    for worker_rank in range(1, size):
                        send_req_partition.append(comm.Isend([task_partition[(worker_rank-1):(worker_rank+1)], MPI.INT],
                                                             dest=worker_rank))
                    MPI.Request.waitall(send_req_partition)

                    # ############# copied from 1_2 #######################
                    for epoch in range(num_epocs_tp_kp):
                        for i, (images, labels) in enumerate(train_loader_CIFAR10):
                            if i == num_steps:
                                break
                            start_iter_time = time.time()
                            # sending train-data shapes to each worker
                            send_reqs_labels_shapes = []
                            send_reqs_images_shapes = []

                            for worker_rank in range(1, size):
                                send_reqs_labels_shapes.append(comm.Isend([np.array(labels.size(), dtype='i'), MPI.INT],
                                                                          dest=worker_rank))
                                send_reqs_images_shapes.append(comm.Isend([np.array(images.size(), dtype='i'), MPI.INT],
                                                                          dest=worker_rank))
                            MPI.Request.waitall([*send_reqs_labels_shapes, *send_reqs_images_shapes])

                            # send new parameters of the model to the workers
                            send_reqs_params = []
                            for worker_rank in range(1, size):
                                for index, param in enumerate(model.parameters()):
                                    send_reqs_params.append(
                                        comm.Isend([np.array(param.data.cpu(), dtype='float'), MPI.FLOAT],
                                                   dest=worker_rank, tag=batch_size + index))

                                    # sending train-data to each worker
                            send_reqs_labels = []
                            send_reqs_images = []

                            for worker_rank in range(1, size):
                                send_reqs_images.append(comm.Isend([np.array(images, dtype='float'), MPI.FLOAT],
                                                                   dest=worker_rank, tag=worker_rank * 2))
                                send_reqs_labels.append(
                                    comm.Isend([np.array(labels, dtype='i'), MPI.INT], dest=worker_rank,
                                               tag=2 * worker_rank + 1))

                            MPI.Request.waitall([*send_reqs_images, *send_reqs_params, *send_reqs_labels])

                            # getting calculated gradients and job vectors from each worker
                            # opening buffers for Irecv
                            grad_buffer_list = []  # for getting job's gradients
                            job_vect_buffer_list = []  # for getting job's vector
                            for worker_rank in range(1, size):
                                for j in range(task_split_vect[worker_rank - 1]):
                                    grad_buffer_list.append(
                                        [np.empty(param.size(), dtype='float') for param in model.parameters()])  # creating buffer
                                    job_vect_buffer_list.append(np.empty([n], dtype='float'))

                            # opening Irecv requests for all possible jobs
                            recv_grad_req = []  # for getting job's gradients
                            recv_job_vect_req = []  # for getting job's vector
                            for worker_rank in range(1, size):
                                for j in range(task_split_vect[worker_rank - 1]):
                                    recv_job_vect_req.append(
                                        comm.Irecv([job_vect_buffer_list[task_partition[worker_rank - 1] + j], MPI.FLOAT],
                                                   source=worker_rank, tag=j))
                                    for index, param in enumerate(model.parameters()):
                                        recv_grad_req.append(comm.Irecv(
                                            [grad_buffer_list[task_partition[worker_rank - 1] + j][index], MPI.FLOAT],
                                            source=worker_rank, tag=n * (index + 1) + j))

                            # initializing finished jobs mat and gradients
                            finished_jobs_mat = np.empty([n, n], dtype='float')
                            finished_jobs_grad_list = []
                            for inx_fin in range(n):
                                finished_jobs_grad_list.append(
                                    [np.empty(param.size(), dtype='float') for param in model.parameters()])

                            # waiting for only k jobs to be finished, and saving their gradients and vectors
                            finished_iter_time = start_iter_time  # only initiation
                            finished_tp_kp_time = np.ones((num_workers,), dtype='float') * start_iter_time
                            for num_finished_job in range(n):
                                index_finished_job, _ = MPI.Request.waitany(recv_job_vect_req)  # returns the finished req index
                                MPI.Request.waitall(recv_grad_req[index_finished_job * num_params:(index_finished_job + 1) * num_params])

                                # sys.stdout.write('step = {}, indx finished job = {}, time since iter start = {}\n'
                                #                     .format(i, index_finished_job, time.time()-start_iter_time))
                                # sys.stdout.flush()

                                finished_jobs_grad_list[num_finished_job][:] = grad_buffer_list[index_finished_job][:]
                                finished_jobs_mat[num_finished_job, :] = job_vect_buffer_list[index_finished_job]

                                if num_finished_job == k - 1:
                                    finished_iter_time = time.time()

                                worker_rank = np.argmax(task_partition - index_finished_job > 0)  # finds the worker rank because np.argmax returns the first True index
                                job_num = index_finished_job - task_partition[worker_rank-1]  # finds the jobs index per worker
                                if job_num == (task_split_vect[worker_rank-1] - 1):
                                    finished_tp_kp_time[worker_rank-1] = time.time()

                            # MPI.Request.waitall(recv_grad_req)
                            
                            # sys.stdout.write('step = {}, Titer = {}\n'.format(i, finished_iter_time-start_iter_time))
                            # sys.stdout.flush()
                            
                            tp_kp_time_vect_list.append(finished_tp_kp_time-start_iter_time)
                            t_iter_list.append(finished_iter_time-start_iter_time)

                    tp_kp_time_array = np.array(tp_kp_time_vect_list, dtype='float')
                    mean_tp_kp_vect = np.mean(tp_kp_time_array, axis=0)
                    var_tp_kp_vect = np.var(tp_kp_time_array, axis=0)
                    second_moment_tp_kp_vect = var_tp_kp_vect + (mean_tp_kp_vect**2)

                    prob_tp_kp_vect = mean_tp_kp_vect + gamma * second_moment_tp_kp_vect
                    mismatch = np.var(prob_tp_kp_vect)

                    mean_time_iter = np.mean(np.array(t_iter_list, dtype='float'))

                    print_to_console('Master:\n mean Tp_kp = {}, var Tp_kp = {}\n mismatch = {}, mean Titer = {}\n'
                                     .format(mean_tp_kp_vect, var_tp_kp_vect, mismatch, mean_time_iter))

                    metadata_list.append(np.array([n, d, k, omega, c, mean_tp_vect, var_tp_vect, mean_cp_vect,
                                                   task_split_vect, mean_tp_kp_vect, var_tp_kp_vect, mismatch,
                                                   mean_time_iter, mean_tp_calc_vect, mean_tp_grad_vect,
                                                   tp_time_array]))

                # ########################copied from 1_2#################################
        df_metadata = pd.DataFrame(metadata_list, columns=['n', 'd', 'k', 'omega', 'c', 'mean_tp_vect', 'var_tp_vect',
                                                           'mean_comm_time_vect', 'task_split_vect', 'mean_tp_kp_vect',
                                                           'var_tp_kp_vect', 'mismatch', 'mean_time_iter',
                                                           'mean_tp_calc_vect', 'mean_tp_grad_vect', 'tp_time_array'])
        df_metadata.to_excel('/home/ubuntu/cloud/metadata_fixed_4.xlsx')

    else:  # ############################################################ Workers ##################################################################
        model.train()
        # ########### part 1: tp estimation #################
        for n in n_vect:
            partition_vect = np.arange(start=0, stop=batch_size, step=(batch_size / n),
                                       dtype='i')  # vector of start indices for each possible calcs
            for d in range(1, n):
                # receive code matrix
                code_matrix = np.empty([n, n], dtype='float')
                recv_req_code_matrix = comm.Irecv([code_matrix, MPI.FLOAT], source=0)
                recv_req_code_matrix.wait()

                for epoch in range(num_epocs_tp):
                    for i in range(num_runs_per_epoc):
                        if i == num_steps:
                            break
                        
                        labels_shape = np.empty([1], dtype='i')
                        images_shape = np.empty([4], dtype='i')

                        labels_shape_req = comm.Irecv([labels_shape, MPI.INT], source=0)
                        images_shape_req = comm.Irecv([images_shape, MPI.INT], source=0)

                        MPI.Request.waitall([labels_shape_req, images_shape_req])

                        # recv new parameters from master and update them in your model
                        recv_param_req = []
                        param_buffer = [np.empty(param.size(), dtype='float') for param in model.parameters()]
                        for index, param in enumerate(model.parameters()):
                            recv_param_req.append(comm.Irecv([param_buffer[index], MPI.FLOAT], source=0,
                                                             tag=batch_size + index))

                        MPI.Request.waitall(recv_param_req)

                        # recv labels and images
                        images = np.empty(images_shape, dtype='float')
                        labels = np.empty(labels_shape, dtype='i')

                        images_req = comm.Irecv([images, MPI.FLOAT], source=0, tag=rank*2)
                        labels_req = comm.Irecv([labels, MPI.INT], source=0, tag=rank*2+1)

                        images_req.wait()
                        labels_req.wait()

                        # get a code vect from the code matrix
                        task_index = epoch * num_runs_per_epoc + i
                        code_index = task_index % n
                        code_vect = code_matrix[code_index, :]

                        partition_vect_per_task = partition_vect[code_vect != 0]  # vector of relevant partition indices
                        code_vec_sq = code_vect[code_vect != 0]  # vector of non-zero coeffs

                        # preforming specific job calculations
                        start_tp_calc_time = time.time()
                        tot_grad_list = [np.zeros(param.size(), dtype='float') for param in
                                         model.parameters()]  # initializing total gradient list
                        for calc_num, index_calc_part in enumerate(partition_vect_per_task):
                            # getting relevant images and labels for a calc from the job
                            images_per_calc = torch.from_numpy(
                                images[index_calc_part:index_calc_part + int(batch_size / n),
                                :, :, :]).float().to(device)
                            labels_per_calc = torch.from_numpy(
                                labels[index_calc_part:index_calc_part + int(batch_size / n)]
                                ).long().to(device)

                            outputs = model(images_per_calc)
                            loss = criterion(outputs, labels_per_calc)
                            optimizer.zero_grad()  # zero (flush) the gradients from the previous iteration
                            loss.backward()  # calculate the gradients w.r.t the loss function (saved in tensor.grad field)

                            # print worker's learning status
                            if calc_num == 0 and (i + 1) % num_print_cycle == 0:
                                print_to_console('Worker {}: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n'
                                                 .format(rank, epoch + 1, num_epocs_tp, i + 1, num_runs_per_epoc,
                                                         loss.item()))

                            # aggregating the gradients of the calc and mul by relevant coefficients
                            for inx_param, param in enumerate(model.parameters()):
                                tot_grad_list[inx_param] += np.array(param.grad.cpu(), dtype='float') * code_vec_sq[
                                    calc_num]
                        end_tp_calc_time = time.time()

                        # send_req_time = comm.Isend([np.array([end_tp_calc_time - start_tp_calc_time], dtype='float'),
                        #                             MPI.FLOAT], dest=0, tag=888888)
                        # send_req_time.wait()

                        send_req_tot_grad_tp = [comm.Isend([tot_grad_list[index], MPI.FLOAT], dest=0, tag=index) for index in
                                                range(num_params)]

                        MPI.Request.waitall(send_req_tot_grad_tp)

                        send_req_time = comm.Isend([np.array([start_tp_calc_time, end_tp_calc_time], dtype='float'),
                                                    MPI.FLOAT], dest=0, tag=888888)
                        send_req_time.wait()

                        # sys.stdout.write('Worker {}: tp grad time = {}\n'.format(rank, time.time()-end_tp_calc_time))
                        # sys.stdout.flush()


                # ########### part 2:optimal load split #################
                is_split_uni = np.empty([1], dtype='i')
                recv_req_is_split_uni = comm.Irecv([is_split_uni, MPI.INT], source=0)
                recv_req_is_split_uni.wait()

                # get task partition indices
                for num_split_run in range(2):
                    if num_split_run == 1 and is_split_uni[0] == 1:
                        break
                    code_mat_partition = np.empty([2], dtype='i')
                    recv_req_partition = comm.Irecv([code_mat_partition, MPI.INT], source=0)
                    recv_req_partition.wait()

                    # get relevant subset of task matrix
                    jobs_mat = code_matrix[code_mat_partition[0]:code_mat_partition[1], :]

                    # ############# copied from 1_2 #######################
                    for epoch in range(num_epocs_tp_kp):
                        for i in range(num_runs_per_epoc):
                            if i == num_steps:
                                break
                            # recv labels and images shapes
                            labels_shape = np.empty([1], dtype='i')
                            images_shape = np.empty([4], dtype='i')

                            labels_shape_req = comm.Irecv([labels_shape, MPI.INT], source=0)
                            images_shape_req = comm.Irecv([images_shape, MPI.INT], source=0)

                            MPI.Request.waitall([labels_shape_req, images_shape_req])

                            # recv new parameters from master and update them in your model
                            recv_param_req = []
                            param_buffer = [np.empty(param.size(), dtype='float') for param in model.parameters()]
                            for index, param in enumerate(model.parameters()):
                                recv_param_req.append(comm.Irecv([param_buffer[index], MPI.FLOAT], source=0,
                                                                 tag=batch_size + index))

                            # recv labels and images
                            images = np.empty(images_shape, dtype='float')
                            labels = np.empty(labels_shape, dtype='i')

                            images_req = comm.Irecv([images, MPI.FLOAT], source=0, tag=rank * 2)
                            labels_req = comm.Irecv([labels, MPI.INT], source=0, tag=rank * 2 + 1)

                            MPI.Request.waitall([*recv_param_req, images_req, labels_req])

                            # extracting all jobs properties
                            k_p = jobs_mat.shape[0]
                            # n = jobs_mat.shape[1]
                            # d = np.sum(jobs_mat[0, :] != 0)
                            partition_vect = np.arange(start=0, stop=batch_size, step=(batch_size / n),
                                                       dtype='i')  # vector of start indices for each possible calcs

                            # calc and send gradients per job
                            # send_job_vect_req = []
                            # send_tot_grad_job_req = []
                            for job_p_inx in range(k_p):

                                # start_job_calc_time = time.time()

                                # extracting specific job properties
                                job_p_vect = jobs_mat[job_p_inx, :]  # vector of coeff for the job
                                partition_vect_per_job = partition_vect[
                                    job_p_vect != 0]  # vector of relevant partition indices
                                job_p_vec_sq = job_p_vect[job_p_vect != 0]  # vector of non-zero coeffs

                                # preforming specific job calculations
                                tot_grad_list = [np.zeros(param.size(), dtype='float') for param in
                                                 model.parameters()]  # initializing total gradient list
                                for calc_num, index_calc_part in enumerate(partition_vect_per_job):
                                    # getting relevant images and labels for a calc from the job
                                    images_per_calc = torch.from_numpy(
                                        images[index_calc_part:index_calc_part + int(batch_size / n),
                                        :, :, :]).float().to(device)
                                    labels_per_calc = torch.from_numpy(
                                        labels[index_calc_part:index_calc_part + int(batch_size / n)]
                                        ).long().to(device)

                                    outputs = model(images_per_calc)
                                    loss = criterion(outputs, labels_per_calc)
                                    optimizer.zero_grad()  # zero (flush) the gradients from the previous iteration
                                    loss.backward()  # calculate the gradients w.r.t the loss function (saved in tensor.grad field)

                                    # print worker's learning status
                                    if calc_num == 0 and job_p_inx == 0 and (i + 1) % num_print_cycle == 0:
                                        print_to_console('Worker {}: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n'
                                                         .format(rank, epoch + 1, num_epocs_tp_kp, i + 1,
                                                                 num_runs_per_epoc, loss.item()))

                                    # aggregating the gradients of the calc and mul by relevant coefficients
                                    for inx_param, param in enumerate(model.parameters()):
                                        tot_grad_list[inx_param] += np.array(param.grad.cpu(), dtype='float') * \
                                                                    job_p_vec_sq[calc_num]
                                # end_job_calc_time = time.time()

                                # sending total gradient of a job and its vector
                                #send_job_vect_req.append(comm.Isend([job_p_vect, MPI.FLOAT], dest=0, tag=job_p_inx))
                                
                                send_job_vect_req = comm.Isend([job_p_vect, MPI.FLOAT], dest=0, tag=job_p_inx)
                                send_job_vect_req.wait()

                                send_tot_grad_job_req = []
                                for index, tot_grad in enumerate(tot_grad_list):
                                    send_tot_grad_job_req.append(
                                        comm.Isend([tot_grad, MPI.FLOAT], dest=0, tag=n * (index + 1) + job_p_inx))
                                MPI.Request.waitall(send_tot_grad_job_req)

                                # send_job_vect_req = comm.Isend([job_p_vect, MPI.FLOAT], dest=0, tag=job_p_inx)
                                # send_job_vect_req.wait()

                                # end_job_send_time = time.time()

                                # sys.stdout.write('Worker {}: job indx = {},  job calc time = {}, job send time = {}\n'
                                #                    .format(rank, job_p_inx, end_job_calc_time - start_job_calc_time, 
                                #                        end_job_send_time - end_job_calc_time))
                                # sys.stdout.flush()

                            # MPI.Request.waitall([*send_job_vect_req, *send_tot_grad_job_req])
                    # ############# copied from 1_2 #######################
