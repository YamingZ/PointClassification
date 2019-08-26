import os

class Parameters():
    def __init__(self):
        self.isRotation = True
        self.restoreModel = False
        self.useSTN = True
        # self.dataDir = '/home/ym/PycharmProjects/TF_GPU/Data/'    #1080ti
        self.dataDir = '/home/zym/PycharmProjects/data/'            #k80
        # self.modelDir = os.path.dirname(os.path.abspath('.')) + '/model/'
        # self.logDir = os.path.dirname(os.path.abspath('.')) + '/log/'
        # self.fileName = '0221_40nn_cheby_2_2_w_55_52_multi_res.txt'
        self.GpuNums = 4

        #fix parameters
        self.samplingType = 'farthest_sampling'
        self.neighborNumber = 40    #40
        self.pointNumber = 512  #1024
        self.outputClassN = 40
        self.chebyshev_1_Order = 3
        self.chebyshev_2_Order = 3
        self.chebyshev_3_Order = 3
        self.keep_prob_1 = 0.9
        self.keep_prob_2 = 0.9
        self.batchSize = 28*self.GpuNums
        self.evalBatchSize = 28
        self.testBatchSize = 1

        self.learningRate = 1e-2
        self.lr_decay_steps = 100
        self.minimum_lr = 0

        self.weight_scaler = 40  # 40
        self.weighting_scheme = 'weighted'  # uniform weighted,uniform
        self.max_epoch = 30
        self.l2_rate = 1e-4 # 8e-6
        self.tmat_rate = 1e-4

        self.input_data_dim = 3
        self.gcn_1_filter_n = 32 # filter number of the first gcn layer
        self.gcn_2_filter_n = 128 # filter number of the second gcn layer
        self.gcn_3_filter_n = 512 # filter number of the third gcn layer
        self.fc_1_n = 512 # fully connected layer dimension

        #multi res parameters
        self.clusterNumberL1 = 100 # layer one convolutional layer's cluster number
        self.nearestNeighborL1 = 6 # nearest neighbor number of each centroid points when performing max pooling in first gcn

        self.clusterNumberL2 = 20  # layer two convolutional layer's cluster number
        self.nearestNeighborL2 = 6 # nearest neighbor number of each centroid points when performing max pooling in second gcn layer


    def info(self):
        print("GPU's number:{}\n".format(self.GpuNums))
        


