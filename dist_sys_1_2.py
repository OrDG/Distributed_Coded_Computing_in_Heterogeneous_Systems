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
    device = torch.device("cpu")

    # create model, send it to device
    model = TwoLayerNet(D_in=28 * 28, H=256, D_out=10).to(device).float()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if rank == 0:  # ######################################### Master ################################################
        # Calculate num of images per worker
        partition_array = [0, 0.3, 0.3, 0.4]
        for index in range(len(partition_array)-1):
            partition_array[index+1] = partition_array[index] + partition_array[index+1]
        index_partition = np.array([int(frac*batch_size) for frac in partition_array])

        sys.stdout.write('Master: partition indexes = ' + str(index_partition) + '\n')
        sys.stdout.flush()

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
                    send_req_mat.append(comm.Isend([np.array(code_matrix[num_jobs_partition[worker_rank-1]:num_jobs_partition[worker_rank], :]
                                                             , dtype='i'), MPI.INT], dest=worker_rank))
                MPI.Request.waitall(send_req_mat)

                # getting calculated gradients from each worker (needs update)
                grad_buffer_list = []
                for worker_rank in range(1, size):
                    grad_buffer_list.append([np.empty(param.size(), dtype='float') for param in model.parameters()]) # creating buffer

                recv_grad_req = []
                for worker_rank in range(1, size):
                    for index, param in enumerate(model.parameters()):
                        recv_grad_req.append(comm.Irecv([grad_buffer_list[worker_rank-1][index], MPI.FLOAT],
                                                        source=worker_rank))

                MPI.Request.waitall(recv_grad_req)

                # aggregating gradients from results and update model parameters (needs update)
                optimizer.zero_grad()
                for index, param in enumerate(model.parameters()):
                    grad_arr = np.zeros(np.array(grad_buffer_list[0][index]).shape, dtype='float')
                    for worker_rank in range(1, size):
                        grad_arr += np.array(grad_buffer_list[worker_rank - 1][index], dtype='float') # * partition_array[worker_rank]
                    param.grad = torch.from_numpy(grad_arr).float()
                optimizer.step()

                # send new parameters of the model to the workers
                send_reqs_params = []
                for worker_rank in range(1, size):
                    for index, param in enumerate(model.parameters()):
                        send_reqs_params.append(comm.Isend([np.array(param.data, dtype='float'), MPI.FLOAT], dest=worker_rank))

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
                jobs_mat = np.empty(jobs_mat_shape, dtype='i')
                jobs_mat_shape_req = comm.Irecv([jobs_mat, MPI.INT], source=0)
                jobs_mat_shape_req.wait()

                # load data on device (needs update)
                images = torch.from_numpy(images).float()
                labels = torch.from_numpy(labels).long()

                images = images.to(device).view(int(labels_shape), -1)  # represent images as column vectors
                labels = labels.to(device)

                # Forward pass (needs update)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()  # zero (flush) the gradients from the previous iteration
                loss.backward()  # calculate the gradients w.r.t the loss function (saved in tensor.grad field)

                # print worker's learning status
                if (i + 1) % 100 == 0:
                    sys.stdout.write('Worker ' + str(rank) + ' : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f} '
                                                         's'.format(epoch + 1, num_epochs, i+1, total_step, loss.item(),
                                                                    time.time() - start_time) + '\n')
                    sys.stdout.flush()

                # get loss gradients and send to the master (needs update)
                send_grad_req = []
                for name, param in model.named_parameters():
                    send_grad = np.array(param.grad, dtype='float')
                    send_grad_req.append(comm.Isend([send_grad, MPI.FLOAT], dest=0))

                MPI.Request.waitall(send_grad_req)

                # recv new parameters from master and update them in your model
                recv_param_req = []
                param_buffer = [np.empty(param.size(), dtype='float') for param in model.parameters()]
                for index, param in enumerate(model.parameters()):
                    recv_param_req.append(comm.Irecv([param_buffer[index], MPI.FLOAT], source=0))

                MPI.Request.waitall(recv_param_req)

                for index, param in enumerate(model.parameters()):
                    param.data = torch.from_numpy(param_buffer[index]).float()
