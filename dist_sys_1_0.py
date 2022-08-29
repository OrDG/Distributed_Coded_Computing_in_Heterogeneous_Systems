#!/usr/bin/env python
"""
Parallel Hello World
"""

from mpi4py import MPI
import sys
import time
import numpy as np
import pandas as pd
import torch
import sklearn

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    '''
    sys.stdout.write("Hi, World! I am process %d of %d on %s.\n" % (rank, size, name))
    sys.stdout.flush()
    '''
    num_iter = 10 ** 4

    N = 10 ** 4

    beta_arr = [0.1, 0.2, 0.3]

    if rank == 0:  # ######################################### Master ################################################
        num_workers = size - 1
        # N, beta_arr, send_time, calc_time, receive_time, Irecv_extra_worker, Irecv_master
        saved_metadata = np.zeros((num_iter, 4+5*num_workers))
        for iteration in range(num_iter):
            saved_metadata[iteration, 0] = N
            saved_metadata[iteration, 1:4] = beta_arr

            # Init
            time_start_comm = np.zeros(num_workers)
            time_comm = np.zeros(num_workers)
            time_start_calc = np.zeros(num_workers)
            time_calc = np.zeros(num_workers)

            data = np.random.rand(N, 2 * N)
            sys.stdout.write('Master: sending tasks to workers\n')
            sys.stdout.flush()

            send_reqs = []
            comm.barrier()  # sync for comm time meas
            for worker_rank in range(1, size):
                time_start_comm[worker_rank - 1] = time.time()
                send_reqs.append(comm.Isend([data, MPI.FLOAT], dest=worker_rank))

            fin_start_send_time = time.time()
            sys.stdout.write('Master: Irecv extra time per worker- ' + str(fin_start_send_time - time_start_comm) + '\n')
            sys.stdout.flush()

            saved_metadata[iteration, 3*num_workers+4:4*num_workers+4] = fin_start_send_time - time_start_comm  # Irecv extra worker

            while not MPI.Request.Testall(send_reqs):
                index_fin = MPI.Request.waitany(send_reqs)
                time_comm[index_fin] = time.time() - time_start_comm[index_fin]
                sys.stdout.write('Master :  worker ' + str(index_fin[0]+1) + ' got task data \n')
                sys.stdout.flush()

            sys.stdout.write('Master: measured sending time - '+str(time_comm) + '\n\n')
            sys.stdout.flush()

            saved_metadata[iteration, 4:num_workers+4] = time_comm

            recv_data = [np.empty([N, N]) for i in range(1, size)]
            sys.stdout.write('Master: waiting for task completion from workers\n')
            sys.stdout.flush()

            recv_reqs = []
            comm.barrier()  # sync for calc time meas
            time_start_calc[:] = time.time()
            for worker_rank in range(1, size):
                recv_reqs.append(comm.Irecv([recv_data[worker_rank-1], MPI.FLOAT], source=worker_rank))

            fin_start_recv_time = time.time()
            sys.stdout.write('Master: Irecv time - ' + str(fin_start_recv_time-time_start_calc) + '\n')
            sys.stdout.flush()

            saved_metadata[iteration, 4*num_workers+4:5*num_workers+4] = fin_start_recv_time - time_start_calc

            while not MPI.Request.Testall(recv_reqs):
                index_fin = MPI.Request.waitany(recv_reqs)
                time_calc[index_fin] = time.time() - time_start_calc[index_fin]

            sys.stdout.write('Master: measured calculation and receiving time - '+str(time_calc) + '\n')
            sys.stdout.flush()

            real_calc_times = [comm.recv(source=worker_rank) for worker_rank in range(1, size)]
            send_back_times = time_calc - real_calc_times
            sys.stdout.write('Master: measured receiving time - ' + str(send_back_times) + '\n')
            sys.stdout.flush()

            saved_metadata[iteration, 1*num_workers+4:2*num_workers+4] = real_calc_times
            saved_metadata[iteration, 2*num_workers+4:3*num_workers+4] = send_back_times

            sys.stdout.write('Master: measured metadata - N, betas, send_time, calc_time, receive_time, '
                             'Irecv_extra_worker, Irecv_master\n')
            sys.stdout.write('Master: measured metadata - ' + str(saved_metadata[iteration, :]) + '\n')
            sys.stdout.flush()

        df = pd.DataFrame(saved_metadata, columns=['N', 'beta_1', 'beta_2', 'beta_3', 'send_time_1', 'send_time_2',
                                                   'send_time_3', 'calc_time_1', 'calc_time_2', 'calc_time_3',
                                                   'receive_time_1', 'receive_time_2', 'receive_time_3',
                                                   'Irecv_worker_1', 'Irecv_worker_2', 'Irecv_worker_3',
                                                   'Irecv_master_1', 'Irecv_master_2', 'Irecv_master_3'])
        df.to_excel('/home/user/cloud/ProjectB/metadata_artificial.xlsx')

    else:  # ########################################### Workers ############################################
        for iteration in range(num_iter):
            data_recv = np.empty([N, 2*N])
            sys.stdout.write('Worker ' + str(rank) + ': waiting for task from master\n')
            sys.stdout.flush()

            comm.barrier()  # sync for comm time meas
            req_rec_from_master = comm.Irecv([data_recv, MPI.FLOAT], source=0)
            req_rec_from_master.wait()

            comm.barrier()  # sync for calc time meas
            time_start_calc = time.time()

            A = data_recv[:, :N]
            B = data_recv[:, N:]
            calc = np.matmul(A, B)

            # rnd_time_task = np.random.exponential(beta_arr[rank-1])
            # time.sleep(rnd_time_task)
            # calc = np.array(data_recv[:, :N])

            time_for_calc = time.time() - time_start_calc

            sys.stdout.write('Worker '+str(rank)+' : time for calculation = ' + str(time_for_calc) + ', sending answer\n')
            sys.stdout.flush()
            req_send_to_master = comm.Isend([calc, MPI.FLOAT], dest=0)
            req_send_to_master.wait()

            comm.send(time_for_calc, dest=0)
