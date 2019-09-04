import os

class Parameters():
    def __init__(self):
        self.useSTN = False
        self.EvalCycle = 2    #   <1Hz
        self.isRotation = True
        self.restoreModel = False
        self.dataDir = '/home/ym/PycharmProjects/TF_GPU/Data/'    #1080ti
        self.ckptDir =  'CheckPoint/'
        # self.dataDir = '/home/zym/PycharmProjects/data/'            #k80


        #fix parameters
        self.weight_scaler = 40  # 40
        self.weighting_scheme = 'weighted'  # uniform weighted,uniform
        self.samplingType = 'farthest_sampling'
        self.chebyshev_1_Order = 5
        self.chebyshev_2_Order = 4
        self.chebyshev_3_Order = 3
        self.keep_prob_1 = 0.85
        self.keep_prob_2 = 0.9

        # parameters of train evaluation test
        self.max_epoch = 50
        self.trainBatchSize = 28
        self.evalBatchSize = 100
        self.testBatchSize = 1

        # leraning rate
        self.learningRate = 2e-2
        self.lr_decay_steps = 150
        self.lr_decay_rate = 0.95
        self.minimum_lr = 0
        self.l2_rate = 1e-4 # 8e-6
        self.tmat_rate = 1e-4

        # dimension of each layer
        self.input_data_dim = 3
        self.gcn_1_filter_n = 32    # filter number of the first gcn layer
        self.gcn_2_filter_n = 68    # filter number of the second gcn layer
        self.gcn_3_filter_n = 128    # filter number of the third gcn layer
        # self.gcn_4_filter_n = 128   # filter number of the fourth gcn layer
        self.fc_1_n = 1024          # fully connected layer dimension
        # self.fc_2_n = 512           # fully connected layer dimension
        self.outputClassN = 40

        # multi res parameters
        self.pointNumber = 512    #1024 # layer one convolution layer's input point number
        self.neighborNumber = 20    #40 # nearest neighbor number of each centroid points when generating graph in first gcn
        self.poolingRange = 8          # nearest neighbor number of each centroid points when performing max pooling in first gcn

        self.clusterNumberL1 = 128      # layer two convolution layer's input point number
        self.nearestNeighborL1 = 20     # nearest neighbor number of each centroid points when generating graph in second gcn
        self.poolingRangeL1 = 8        # nearest neighbor number of each centroid points when performing max pooling in second gcn

        self.clusterNumberL2 = 32       # layer three convolution layer's input number
        self.nearestNeighborL2 = 20     # nearest neighbor number of each centroid points when generating graph in third gcn layer
        self.poolingRangeL2 = 4        # nearest neighbor number of each centroid points when performing max pooling in third gcn

        self.clusterNumberL3 = 16       # layer four convolution layer's input number
        # self.nearestNeighborL3 = 16     # nearest neighbor number of each centroid points when generating graph in fourth gcn


    def info(self):
        print('\n'.join(['%s: %s' % item for item in self.__dict__.items()]))

    def log(self):
        file = open(self.ckptDir+'hyperparameters.txt','w+')
        file.write('\n'.join(['%s: %s' % item for item in self.__dict__.items()]))


if __name__ == "__main__":
    para = Parameters()
    para.info()
        


