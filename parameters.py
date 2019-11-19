import json

class Parameters():
    def __init__(self,roadparam = False):
        #Hardware Attr
        self.host = "1080ti"

        #FilePath Attr
        self.dataDir = '/home/ym/PycharmProjects/TF_GPU/Data/' if self.host=="1080ti" else '/home/zym/PycharmProjects/data/'#k80
        self.ckptDir = 'CheckPoint/'
        self.evalDir = 'EvaluateData/'
        self.restoreModel = False

        #InputData Attr
        self.weight_scaler = 40  # 40
        self.input_data_dim = 3
        self.samplingType = 'farthest_sampling'
        self.weighting_scheme = 'weighted'  # uniform weighted,uniform
        self.isRotationTrain = True
        self.isRotationEval = True
        self.useSphericalPos = True

        #Model Attr
        self.useSTN = True
        self.useChannelAttention = False
        self.gcn_1_filter_n = 32    # filter number of the first gcn layer
        self.gcn_2_filter_n = 64    # filter number of the second gcn layer
        self.gcn_3_filter_n = 128   # filter number of the third gcn layer
        self.fc_1_n = 2048          # fully connected layer dimension
        self.fc_2_n = 1024          # fully connected layer dimension
        self.fc_3_n = 512           # fully connected layer dimension
        self.outputClassN = 40
        self.keep_prob_1 = 0.9
        self.keep_prob_2 = 0.9

        #GraphConv Attr
        self.chebyshev_1_Order = 3
        self.chebyshev_2_Order = 3
        self.chebyshev_3_Order = 3
        self.chebyshev_4_Order = 3
        self.vertexNumG1 = 512
        self.vertexNumG2 = 256
        self.vertexNumG3 = 64
        self.vertexNumG4 = 32
        self.edgeNumG1 = 20
        self.edgeNumG2 = 20
        self.edgeNumG3 = 20
        self.poolNumG1 = 4
        self.poolNumG2 = 4
        self.poolNumG3 = 2
        self.poolRangeG1 = 0.08
        self.poolRangeG2 = 0.2
        self.poolRangeG3 = 0.5

        #TrainParam Attr
        self.max_epoch = 30
        self.trainBatchSize = 40    #total = 9840
        self.learningRate = 0.001   #1e-2
        self.lr_decay_steps = 150
        self.lr_decay_rate = 0.9
        self.minimum_lr = 0
        self.l2_rate = 0.001        #1e-4
        self.tmat_rate = 0.001

        #EvalParam Attr
        self.EvalCycle = 1          # <1Hz
        self.evalBatchSize = 137    #total = 2468

        #TestParam Attr
        self.testBatchSize = 1

        if roadparam:
            self.load()

    def info(self):
        print('\n'.join(['%s: %s' % item for item in self.__dict__.items()]))

    def save(self):
        file_path = self.ckptDir+"hyperParameters.json"
        json_data = json.dumps(vars(self),sort_keys=False, indent=2)
        with open(file_path,'w') as f:
            f.write(json_data)

    def load(self):
        file_path = self.ckptDir+"hyperParameters.json"
        with open(file_path,'r') as f:
            json_data = json.load(f)
            for name,value in json_data.items():
                if name in vars(self).keys():
                    setattr(self, name, value)

if __name__ == "__main__":
    para = Parameters()
    para.info()
    para.load()
    para.info()


