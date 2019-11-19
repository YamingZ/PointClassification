from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
import numpy as np
import scipy
import utils

class DataSets(object):
    def __init__(self, data, rotate=True, shuffle = True, jitter = True, spherical = True, one_hot_label = True):
        coor, graph, label = data
        self._coor = np.concatenate([value for value in coor.values()])
        self._label = np.concatenate([value for value in label.values()])
        if one_hot_label:
            self._label = label_binarize(self._label,classes=[j for j in range(40)])
        self._graph = scipy.sparse.vstack(list(graph.values())).tocsr()
        self._count = 0
        self._N = len(self._label)

        self.rotate = rotate
        self.shuffle = shuffle
        self.jitter = jitter
        self.spherical = spherical

    def get_all_data(self):
        return self._coor, self._graph, self._label

    def get_samples(self, N=100):
        pass

    def iter(self, batch_size=1):
        if self.shuffle:
            self._coor, self._graph, self._label = shuffle(self._coor, self._graph, self._label)
        '''Return an iterator which iterates on the elements of the dataset.'''
        return self.__iter__(batch_size)

    def __iter__(self, batch_size=1):
        while self._count < np.floor(self._N / batch_size):
            start = self._count * batch_size
            end = start + batch_size
            batchCoor, batchGraph, batchLabel = utils.get_mini_batch(self._coor, self._graph, self._label, start, end)
            self._count += 1
            # data process
            batchGraph = batchGraph.todense()
            if self.jitter:
                batchCoor = utils.jitter_point_cloud(batchCoor)
            if self.rotate:
                batchCoor = utils.rotate_point_cloud(batchCoor)
            if self.spherical:
                batchSCoor = utils.get_Spherical_coordinate(batchCoor,normalized=True)
                yield batchSCoor, batchCoor, batchGraph, batchLabel
            else:
                yield batchCoor, batchGraph, batchLabel
        raise StopIteration

    @property
    def N(self):
        '''Number of elements in the dataset.'''
        return self._N
    @property
    def counter(self):
        return self._count


if __name__ =='__main__':
    import read_data
    from parameters import *


    para = Parameters()
    inputTrain, trainLabel, inputTest, testLabel = read_data.load_data(para.VertexNumG1, para.samplingType,
                                                                       para.dataDir)
    scaledLaplacianTrain, scaledLaplacianTest = read_data.prepareData(inputTrain, inputTest, para.neighborNumber,
                                                                      para.pointNumber, para.dataDir)
    # data = inputTest,scaledLaplacianTest,testLabel
    data = inputTrain, scaledLaplacianTrain, trainLabel
    weight_dict = utils.train_weight_dict(trainLabel, para)

    for value in weight_dict.values():
        print(1/value)

    dataset = DataSets(data)
    print(dataset.N)
    batchSize = 28
    iters = dataset.iter(batchSize)
    batchSCoor, batchCoor, batchGraph, batchLabel = next(iters)

    while True:
        try:
            batchSCoor, batchCoor, batchGraph, batchLabel = next(iters)
            print(batchLabel)
            print(dataset.counter)
        except StopIteration:
            print('data over')
            break