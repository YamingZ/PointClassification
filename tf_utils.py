import numpy as np
import tensorflow as tf
# from sampling.tf_sampling import *
# from HDF5_loader import *

# make no sense
# def farthest_sampling(batch_original_coor, batch_size, nodes_n, M, k):
#     # input    1) batch_original_coor:  a.coordinate (B,N,3)
#     #                                   b.input features (B,N,n1)
#     #          2) batch_size = B
#     #          3) nodes_n = N
#     #          4) M centroid point number(cluster number)
#     #          5) k nearest neighbor number
#     # output:  1) batch index (B, M*k,1)
#     #          2) centroid points (B, M, 3)
#     centroid_points_ids = farthest_point_sample(M, batch_original_coor)    #(B,M,1)
#     centroid_points = gather_point(batch_original_coor, centroid_points_ids)  #(B,M,3)
#     batch_local_points_id = []
#     for batch_index in range(batch_size):
#         original_coor = batch_original_coor[batch_index]
#         centroid_points_id = centroid_points_ids[batch_index]
#         # calculate distance between each centroid_points and others
#         M_indices = []
#         for i in range(M):
#             centroid_point_id = centroid_points_id[i]
#             centroid_point = original_coor[centroid_point_id,:] #[x,y,z]
#             # (X - Y)*(X - Y) = -2X*Y + X*X + Y*Y
#             d1 = -2 * tf.matmul([centroid_point], tf.transpose(original_coor,[1,0]))
#             d2 = tf.reduce_sum(tf.square([centroid_point]),axis=1,keepdims=True)
#             d3 = tf.reduce_sum(tf.square(original_coor),axis=1)
#             distances = tf.sqrt(d1+d2+d3)
#             _,indices = tf.nn.top_k(-distances,k,sorted=True)
#             M_indices.append(indices)
#         local_points_id = tf.concat(M_indices,axis=0)
#         local_points_id = tf.reshape(local_points_id,[M*k])
#         batch_local_points_id.append([local_points_id])
#     batch_local_id = tf.concat(batch_local_points_id,axis=0)
#     return batch_local_id,centroid_points
#
# def non_loop(test_matrix, train_matrix):
#     num_test = test_matrix.shape[0]
#     num_train = train_matrix.shape[0]
#     dists = np.zeros((num_test, num_train))
#
#     d1 = -2 * np.dot(test_matrix, train_matrix.T)    # shape (num_test, num_train)
#     d2 = np.sum(np.square(test_matrix), axis=1, keepdims=True)    # shape (num_test, 1)
#     d3 = np.sum(np.square(train_matrix), axis=1)     # shape (num_train, )
#     dists = np.sqrt(d1 + d2 + d3)  # broadcasting
#     return dists


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.每个变量以及该变量的梯度
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)
      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


if __name__ == '__main__':
    h5 = HDF5Data('/home/ym/PycharmProjects/TF_GPU/Data/modelnet/modelnet40_ply_hdf5_2048/ply_data_train4.h5')
    batch_points = np.array(h5.getValue('data'))[0:2,0:1024]
    # batch_points = np.reshape(batch_points,[])
    print(batch_points.shape)
    coordinate = tf.placeholder(tf.float32, [None, 1024, 3], name='coordinate')
    sampling_id,sampling_data = farthest_sampling(coordinate,2,1024,50,10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {coordinate:batch_points}
        id,data = sess.run([sampling_id,sampling_data],feed_dict=feed_dict)
        print(id.shape)
        print(data.shape)