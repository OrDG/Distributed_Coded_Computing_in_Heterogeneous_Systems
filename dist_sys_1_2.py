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
import code_matrix_gen as genb


# define a two-layer MLP
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        Parameters:
        D_in - dimensions of inputs
        H - number of hidden units per layer
        D_out - dimensions of outputs
        """
        # initializing the parent object (important!)
        super(TwoLayerNet, self).__init__()
        # define the first layer (hidden)
        self.linear1 = torch.nn.Linear(D_in, H)
        # define the second layer (output)
        self.linear2 = torch.nn.Linear(H, D_out)
        # define the activation function
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        Parameters:
        x - tensor of inputs (shape: [BATCH_SIZE, D_in])
        """
        h_relu = self.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    num_workers = size - 1

    # hyper-parameters:
    batch_size = 256
    num_epochs = 10
    learning_rate = 0.001
    size_dataset = 60000

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create model, send it to device
    model = TwoLayerNet(D_in=28 * 28, H=256, D_out=10).to(device).float()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if rank == 0:  # ######################################### Master ################################################
        # Get  MNIST dataset
        mnist_train_dataset = torchvision.datasets.MNIST(root='./datasets/',
                                                         train=True,
                                                         transform=torchvision.transforms.ToTensor(),
                                                         download=True)

        # Get data loader for train-set
        mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=True, drop_last=True)
        # We use drop_last=True to avoid the case where the data / batch_size != int

        # Get code matrix
        d = 4
        n = 8
        code_matrix = genb.get_rnd_b_mat(d, n)
        k, omega, c = genb.get_code_coeff(code_matrix)

        sys.stdout.write('Master: code matrix = \n{}\n k = {}, omega = {}, c = {}\n'.format(code_matrix, k, omega, c))
        sys.stdout.flush()

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(mnist_train_loader):

                # sending train-data shapes to each worker
                send_reqs_labels_shapes = []
                send_reqs_images_shapes = []

                for worker_rank in range(1, size):
                    send_reqs_labels_shapes.append(comm.Isend([np.array(labels.size(), dtype='i'), MPI.INT],
                                                              dest=worker_rank))
                    send_reqs_images_shapes.append(comm.Isend([np.array(images.size(), dtype='i'), MPI.INT],
                                                              dest=worker_rank))
                MPI.Request.waitall([*send_reqs_labels_shapes, *send_reqs_images_shapes])

                # sending train-data to each worker
                send_reqs_labels = []
                send_reqs_images = []

                for worker_rank in range(1, size):
                    send_reqs_labels.append(comm.Isend([np.array(labels, dtype='i'), MPI.INT], dest=worker_rank))
                    send_reqs_images.append(comm.Isend([np.array(images, dtype='float'), MPI.FLOAT], dest=worker_rank))
                MPI.Request.waitall([*send_reqs_labels, *send_reqs_images])

                # calculating job distribution to each worker, according to a jobs array
                num_jobs_array = np.array([2, 3, 3], dtype='i')  # needs to be in size of num_workers and sum up to n
                num_jobs_partition = np.array([np.sum(num_jobs_array[:j]) for j in range(num_workers + 1)])

                # sending job matrix shapes to each worker
                send_req_mat_shapes = []
                for worker_rank in range(1, size):
                    send_req_mat_shapes.append(comm.Isend([np.array([num_jobs_array[worker_rank-1], n], dtype='i'),
                                                           MPI.INT], dest=worker_rank))

                MPI.Request.waitall(send_req_mat_shapes)

                # sending job matrix to each worker
                send_req_mat = []
                for worker_rank in range(1, size):
                    send_req_mat.append(comm.Isend([np.array(code_matrix[num_jobs_partition[worker_rank-1]:
                                                                         num_jobs_partition[worker_rank], :]
                                                             , dtype='float'), MPI.FLOAT], dest=worker_rank))
                MPI.Request.waitall(send_req_mat)

                # getting calculated gradients and job vectors from each worker
                # opening buffers for Irecv
                grad_buffer_list = []  # for getting job's gradients
                job_vect_buffer_list = []  # for getting job's vector
                for worker_rank in range(1, size):
                    for j in range(num_jobs_array[worker_rank-1]):
                        grad_buffer_list.append([np.empty(param.size(), dtype='float') for param in model.parameters()])  # creating buffer
                        job_vect_buffer_list.append(np.empty([n], dtype='float'))

                # opening Irecv requests for all possible jobs
                recv_grad_req = []  # for getting job's gradients
                recv_job_vect_req = []  # for getting job's vector
                for worker_rank in range(1, size):
                    for j in range(num_jobs_array[worker_rank-1]):
                        recv_job_vect_req.append(comm.Irecv([job_vect_buffer_list[num_jobs_partition[worker_rank-1]+j], MPI.FLOAT],
                                                            source=worker_rank, tag=j))
                        for index, param in enumerate(model.parameters()):
                            recv_grad_req.append(comm.Irecv([grad_buffer_list[num_jobs_partition[worker_rank-1]+j][index], MPI.FLOAT],
                                                            source=worker_rank, tag=n*(index+1)+j))

                # initializing finished jobs mat and gradients
                finished_jobs_mat = np.empty([k, n], dtype='float')
                finished_jobs_grad_list = []
                for inx_fin in range(k):
                    finished_jobs_grad_list.append([np.empty(param.size(), dtype='float') for param in model.parameters()])

                # waiting for only k jobs to be finished, and saving their gradients and vectors
                num_params = len(list(model.parameters()))
                for num_finished_job in range(k):
                    index_finished_job, _ = MPI.Request.waitany(recv_job_vect_req)  # returns the finished req index
                    MPI.Request.waitall(recv_grad_req[index_finished_job * num_params:(index_finished_job+1) * num_params])

                    finished_jobs_grad_list[num_finished_job][:] = grad_buffer_list[index_finished_job][:]
                    finished_jobs_mat[num_finished_job, :] = job_vect_buffer_list[index_finished_job]

                    # worker_rank = np.argmax(num_jobs_partition - index_finished_job > 0)  # finds the worker rank because np.argmax returns the first True index
                    # MPI.Request.waitall(recv_grad_req[num_jobs_partition[worker_rank-1]*num_params:num_jobs_partition[worker_rank]*num_params])
                    # job_num = index_finished_job - num_jobs_partition[worker_rank-1]  # finds the jobs index per worker

                # sending message that the iteration is finished and cancel all left calcs and reqs
                finished_iter_req = []
                for worker_rank in range(1, size):
                    finished_iter_req.append(comm.Isend([np.zeros([1]), MPI.INT], dest=worker_rank, tag=999999))
                MPI.Request.waitall(finished_iter_req)

                # canceling all recv req
                for recv_req in recv_job_vect_req:
                    if not MPI.Request.Test(recv_req):
                        MPI.Request.Cancel(recv_req)

                for recv_req in recv_grad_req:
                    if not MPI.Request.Test(recv_req):
                        MPI.Request.Cancel(recv_req)

                # calculating coefficients for gradient aggregation
                coeff_vect = genb.calc_agg_coeff(finished_jobs_mat)

                # aggregating gradients from results and update model parameters
                optimizer.zero_grad()
                for index, param in enumerate(model.parameters()):
                    grad_arr = np.zeros(finished_jobs_grad_list[0][index].shape, dtype='float')
                    for job_inx in range(k):
                        grad_arr += finished_jobs_grad_list[job_inx][index] * coeff_vect[job_inx]
                    param.grad = torch.from_numpy(grad_arr).float().to(device)
                optimizer.step()

                # send new parameters of the model to the workers
                send_reqs_params = []
                for worker_rank in range(1, size):
                    for index, param in enumerate(model.parameters()):
                        send_reqs_params.append(comm.Isend([np.array(param.data.cpu(), dtype='float'), MPI.FLOAT], dest=worker_rank))

                MPI.Request.waitall(send_reqs_params)


    else:  # ########################################### Workers ############################################
        model.train()
        start_time = time.time()
        total_step = int(np.floor(size_dataset/batch_size))
        for epoch in range(num_epochs):
            for i in range(total_step):
                # recv labels and images shapes
                labels_shape = np.empty([1], dtype='i')
                images_shape = np.empty([4], dtype='i')

                labels_shape_req = comm.Irecv([labels_shape, MPI.INT], source=0)
                images_shape_req = comm.Irecv([images_shape, MPI.INT], source=0)

                MPI.Request.waitall([labels_shape_req, images_shape_req])

                # recv labels and images
                labels = np.empty(labels_shape, dtype='i')
                images = np.empty(images_shape, dtype='float')

                labels_req = comm.Irecv([labels, MPI.INT], source=0)
                images_req = comm.Irecv([images, MPI.FLOAT], source=0)

                MPI.Request.waitall([labels_req, images_req])

                # recv jobs matrix shape
                jobs_mat_shape = np.empty([2], dtype='i')
                jobs_mat_shape_req = comm.Irecv([jobs_mat_shape, MPI.INT], source=0)
                jobs_mat_shape_req.wait()

                # recv jobs matrix
                jobs_mat = np.empty(jobs_mat_shape, dtype='float')
                jobs_mat_req = comm.Irecv([jobs_mat, MPI.FLOAT], source=0)
                jobs_mat_req.wait()

                # opening finish req
                recv_fin_iter_req = comm.Irecv([np.empty([1]), MPI.INT], source=0, tag=999999)

                # extracting all jobs properties
                k_p = jobs_mat.shape[0]
                n = jobs_mat.shape[1]
                d = np.sum(jobs_mat[0, :] != 0)
                partition_vect = np.arange(start=0, stop=batch_size, step=(batch_size/n), dtype='i')  # vector of start indices for each possible calcs

                # calc and send gradients per job
                send_job_vect_req = []
                send_tot_grad_job_req = []
                for job_p_inx in range(k_p):

                    # extracting specific job properties
                    job_p_vect = jobs_mat[job_p_inx, :]  # vector of coeff for the job
                    partition_vect_per_job = partition_vect[job_p_vect != 0]  # vector of relevant partition indices
                    job_p_vec_sq = job_p_vect[job_p_vect != 0]  # vector of non-zero coeffs

                    # preforming specific job calculations
                    tot_grad_list = [np.zeros(param.size(), dtype='float') for param in model.parameters()]  # initializing total gradient list
                    for calc_num, index_calc_part in enumerate(partition_vect_per_job):
                        # getting relevant images and labels for a calc from the job
                        images_per_calc = torch.from_numpy(images[index_calc_part:index_calc_part + int(batch_size/n),
                                                           :, :, :]).float().to(device).view(int(batch_size/n), -1)
                        labels_per_calc = torch.from_numpy(labels[index_calc_part:index_calc_part + int(batch_size/n)]
                                                           ).long().to(device)

                        outputs = model(images_per_calc)
                        loss = criterion(outputs, labels_per_calc)
                        optimizer.zero_grad()  # zero (flush) the gradients from the previous iteration
                        loss.backward()  # calculate the gradients w.r.t the loss function (saved in tensor.grad field)

                        # print worker's learning status
                        if calc_num == 0 and job_p_inx == 0 and (i + 1) % 50 == 0:
                            sys.stdout.write(
                                'Worker ' + str(rank) + ' : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f} '
                                                        's'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                   loss.item(), time.time() - start_time) + '\n')
                            sys.stdout.flush()

                        # aggregating the gradients of the calc and mul by relevant coefficients
                        for inx_param, param in enumerate(model.parameters()):
                            tot_grad_list[inx_param] += np.array(param.grad.cpu(), dtype='float') * job_p_vec_sq[calc_num]

                    # sending total gradient of a job and its vector
                    send_job_vect_req.append(comm.Isend([job_p_vect, MPI.FLOAT], dest=0, tag=job_p_inx))
                    for index, tot_grad in enumerate(tot_grad_list):
                        send_tot_grad_job_req.append(comm.Isend([tot_grad, MPI.FLOAT], dest=0, tag=n*(index+1)+job_p_inx))

                    # checking if finished iteration message was sent, and breaking if so
                    if recv_fin_iter_req.Test():
                        break

                # if finished all jobs before end of iter, wait for end of iter
                if not recv_fin_iter_req.Test():
                    recv_fin_iter_req.wait()

                # canceling all send req
                for send_req in send_job_vect_req:
                    MPI.Request.Cancel(send_req)

                for send_req in send_tot_grad_job_req:
                    MPI.Request.Cancel(send_req)

                # recv new parameters from master and update them in your model
                recv_param_req = []
                param_buffer = [np.empty(param.size(), dtype='float') for param in model.parameters()]
                for index, param in enumerate(model.parameters()):
                    recv_param_req.append(comm.Irecv([param_buffer[index], MPI.FLOAT], source=0))

                MPI.Request.waitall(recv_param_req)

                for index, param in enumerate(model.parameters()):
                    param.data = torch.from_numpy(param_buffer[index]).float().to(device)
