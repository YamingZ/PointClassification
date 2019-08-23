#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:53:31 2017

@author: yingxuezhang
"""
import h5py
import numpy as np
from numpy import linalg as LA
import scipy
from scipy.spatial import cKDTree
import sklearn.metrics
import random
from scipy.spatial.distance import cdist

def getDataFiles(list_filename):
    BASE_DIR = "/".join((list_filename.split('/'))[0:-1])
    return [BASE_DIR+'/'+line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0
    # Weights.
    sigma2 = np.mean(dist[:, -1]) ** 2
    #print sigma2
    dist = np.exp(- dist ** 2 / sigma2)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M * k)
    V = dist.reshape(M * k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))
    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    return W

def normalize_adj(adj):
    adj = scipy.sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalized_laplacian(adj):
    adj_normalized = normalize_adj(adj)
    norm_laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    return norm_laplacian

def scaled_laplacian(adj):
    adj_normalized = normalize_adj(adj)
    laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = scipy.sparse.linalg.eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.sparse.eye(adj.shape[0])
    return scaled_laplacian

def scaled_laplacian_appx(adj):
    adj_normalized = normalize_adj(adj)
    laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    scaled_laplacian = laplacian - scipy.sparse.eye(adj.shape[0])
    return scaled_laplacian


def get_mini_batch(x_signal,graph,y, start, end):
    return x_signal[start:end],graph[start:end],y[start:end]

def add_noise(batch_data,sigma=0.015,clip=0.05):
    batch_n,nodes_n_1,feature_n=batch_data.shape
    noise=np.clip(sigma*np.random.randn(batch_n,nodes_n_1,feature_n),-1*clip,clip)
    new_data=batch_data+noise
    return new_data

def weight_dict_fc(trainLabel, para):
    train_labels = []
    for i in range(len(trainLabel)):
        [train_labels.append(j) for j in trainLabel[i]]
    from sklearn.preprocessing import label_binarize
    y_total_40=label_binarize(train_labels, classes=[i for i in range(40)])
    class_distribution_40_class=np.sum(y_total_40,axis=0)
    class_distribution_40_class=[float(i) for i in class_distribution_40_class]
    class_distribution_40_class=class_distribution_40_class/np.sum(class_distribution_40_class)
    inverse_dist=1/class_distribution_40_class
    norm_inv_dist=inverse_dist/np.sum(inverse_dist)
    weights=norm_inv_dist*para.weight_scaler+1
    weight_dict = dict()
    for classID, value in enumerate(weights):
        weight_dict.update({classID: value})
    return weight_dict

def weights_calculation(batch_labels,weight_dict):
    weights = []
    batch_labels = np.argmax(batch_labels,axis =1)
   
    for i in batch_labels:
        weights.append(weight_dict[i])
    return weights

def uniform_weight(trainLabel):
    weights = []
    [weights.append(1) for i in range(len(trainLabel))] #每种类别的点云数据占比均为1
    return weights

def farthest_sampling(batch_original_coor, M, k, batch_size, nodes_n):
    # input    1) coordinate (B,N*3) 2) input features B*N*n1
    #          3)M centroid point number(cluster number) 4) k nearest neighbor number
    # output:  1) batch index (B, M*k)
    #          2) centroid points (B, M*3)
    batch_object_coor = batch_original_coor#.reshape([batch_size, nodes_n, 3])  # (28,1024,3)
    batch_index = np.zeros([batch_size, M * k])
    batch_centroid_points = np.zeros([batch_size, M * 3])
    for j in range(batch_size):
        pc_object_coor = batch_object_coor[j]
        # calculate pair wise distance
        d = sklearn.metrics.pairwise.pairwise_distances(pc_object_coor, metric='euclidean')
        solution_set = []
        remaining_set = [i for i in range(len(d))]
        a = random.randint(0, len(d) - 1)
        solution_set.append(a)
        remaining_set.remove(a)
        # The mechanism of finding the next centroid point is calculate all the distance between remaining
        # points with the existing centroid point and pick the one with the max min value among them
        for i in range(M - 1):
            distance = d[solution_set, :]
            d_r_s = distance[:, remaining_set]
            a = np.min(d_r_s, axis=0)
            max_index = np.argmax(a)
            remain_index = remaining_set[max_index]
            new_index = remain_index
            solution_set.append(new_index)
            remaining_set.remove(new_index)

        select_coor = pc_object_coor[solution_set]
        tree = cKDTree(pc_object_coor)
        dd, ii = tree.query(select_coor, k=k)
        index_select = ii.flatten()
        batch_centroid_points[j] = select_coor.flatten()
        batch_index[j] = index_select

    return batch_index, batch_centroid_points

def farthest_sampling_new(batch_original_coor, M, k, batch_size, nodes_n):
    # input    1) coordinate (B,N*3) 2) input features B*N*n1
    #          3)M centroid point number(cluster number) 4) k nearest neighbor number
    # output:  1) batch index (B, M*k)
    #          2) centroid points (B, M*3)
    batch_object_coor = batch_original_coor.reshape([batch_size, nodes_n, 3])  # (28,1024,3)
    batch_index = np.zeros([batch_size, M * k])
    batch_centroid_points = np.zeros([batch_size, M * 3])
    for j in range(batch_size):
        pc_object_coor = batch_object_coor[j]
        # calculate pair wise distance
        random.seed(1)
        initial_index = random.randint(0, nodes_n-1)
        initial_point = pc_object_coor[initial_index]
        initial_point = initial_point[np.newaxis,:]
        
        distance = np.zeros((M, nodes_n))
        distance[0] = cdist(initial_point, pc_object_coor)
        solution_set = []
        remaining_set = [i for i in range(nodes_n)]
        a = random.randint(0, nodes_n - 1)
        solution_set.append(a)
        remaining_set.remove(a)
        
        # The mechanism of finding the next centroid point is calculate all the distance between remaining
        # points with the existing centroid point and pick the one with the max min value among them
        for i in range(M - 1):
            d_r_s = distance[0:i+1,:]
            a = np.min(d_r_s, axis=0)
            max_index = np.argmax(a)
            solution_set.append(max_index)
            new_coor = pc_object_coor[max_index]
            new_coor = new_coor[np.newaxis,:]
            d = cdist(new_coor, pc_object_coor)
            distance[i+1] = d
        
        select_coor = pc_object_coor[solution_set]
        tree = cKDTree(pc_object_coor)
        dd, ii = tree.query(select_coor, k=k)
        index_select = ii.flatten()
        batch_centroid_points[j] = select_coor.flatten()
        batch_index[j] = index_select
    return batch_index, batch_centroid_points



def middle_graph_generation(centroid_coordinates, batch_size, M):
    # (1)input:
    #   centroid coordinates (B,M*3)
    # (2)output:
    #   batch graph (B,M*M) in sparse matrix format
    centroid_coordinates = centroid_coordinates.reshape(batch_size, M, 3)
    batch_middle_graph = np.zeros([batch_size, M * M])
    for i in range(len(centroid_coordinates)):
        select_coor = centroid_coordinates[i]
        tree = cKDTree(select_coor)
        dd, ii = tree.query(select_coor, k=M-5) #M-5 #40 #55
        A = adjacency(dd, ii)
        L_scaled = scaled_laplacian(A).todense()
        L_scaled = np.array(L_scaled).flatten()
        batch_middle_graph[i] = L_scaled
    return batch_middle_graph

def shuffle_data(data,graph,labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...],graph[idx,...],labels[idx]


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle_y = np.random.uniform() * 2 * np.pi
        rotation_angle_x = np.random.uniform() * 2 * np.pi
        rotation_angle_z = np.random.uniform() * 2 * np.pi
        cosval_y = np.cos(rotation_angle_y)
        sinval_y = np.sin(rotation_angle_y)
        rotation_matrix_y = np.array([[cosval_y, 0, sinval_y],
                                      [0, 1, 0],
                                      [-sinval_y, 0, cosval_y]])

        cosval_x = np.cos(rotation_angle_x)
        sinval_x = np.sin(rotation_angle_x)
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, cosval_x, -sinval_x],
                                      [0, sinval_x, cosval_x]])

        cosval_z = np.cos(rotation_angle_z)
        sinval_z = np.sin(rotation_angle_z)
        rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
                                      [sinval_z, cosval_z, 0],
                                      [0, 0, 1]])
        rotation_matrix = np.dot(np.dot(rotation_matrix_x,rotation_matrix_y),rotation_matrix_z)
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval_y = np.cos(rotation_angle)
        sinval_y = np.sin(rotation_angle)
        rotation_matrix_y = np.array([[cosval_y, 0, sinval_y],
                                    [0, 1, 0],
                                    [-sinval_y, 0, cosval_y]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix_y)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.008, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def add_rotation(batch_data):
    rotation_Theta = np.random.random()     #[0,PI]
    rotation_Phi = np.random.random()       #[0,2PI]
    print(rotation_Theta,rotation_Phi)
    rotation = [rotation_Theta,rotation_Phi]
    rotation += batch_data
    return rotation

def normalized_point_cloud(batch_data):
    for k in range(batch_data.shape[0]):
        coor = batch_data[k,...]
        coor_min = coor.min()
        coor_max = coor.max()
        coor_nor = ((coor - coor_min)/(coor_max-coor_min)-0.5)*2
        return coor_nor

def get_Spherical_coordinate(batch_data):
    R = LA.norm(batch_data,ord=2,axis=2)
    x = batch_data[:,:,0]
    y = batch_data[:,:,1]
    z = batch_data[:,:,2]
    Theta = np.arctan(y/x)
    Phi = np.arccos(z/R)
    s_coor = np.stack([R,Theta,Phi],axis=2)
    return s_coor

if __name__ == '__main__':
    coor = np.array([[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]]])
    # s_coor = get_Spherical_coordinate(coor)
    # r_s_coor = add_rotation(s_coor)
    print(coor)
    # print(r_s_coor)


