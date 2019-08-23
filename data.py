import tensorflow as tf
import numpy as np
import h5py as h5
from parameters import *

class h5_generator:
    def __init__(self, file):
        self.file = file
    def __call__(self):
        with h5.File(self.file, 'r') as datas:
            for data in datas['data']:
                yield data
            for label in datas['label']:
                yield label

# def h5_generator(file):
#     with h5.File(file, 'r') as datas:
#         for data in datas['data']:
#             yield data

# class h5_generator(object):
#     def __init__(self, file,max):
#         self.max = max
#         self.n, self.a, self.b = 0, 0, 1
#
#     def __iter__(self):
#         return self
#
#     def next(self):
#         if self.n < self.max:
#             r = self.b
#             self.a, self.b = self.b, self.a + self.b
#             self.n = self.n + 1
#             return r
#         raise StopIteration()

class Data(object):
    def __init__(self):
        self.dataset= None
        self.iterator = self.dataset.make_initializable_iterator()

    def read_data(self,file_paths):
        self.dataset=tf.data.Dataset.from_tensor_slices(file_paths)
        self.dataset = self.dataset.interleave(
            lambda filename: tf.data.Dataset.from_generator(
                h5_generator(filename),
                tf.uint8,tf.TensorShape([427,561,3])
            )
        )


    def process_data(self):
        pass

    def get_next_batch(self):
        pass

if __name__ =='__main__':
    para = Parameters()
    datas = h5_generator(para.dataDir+"modelnet/modelnet40_ply_hdf5_2048/ply_data_test0.h5")
    for data in datas():
        print(data.shape)