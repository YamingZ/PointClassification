import utils

class Batch(object):
    def __init__(self,model,batchsize,placeholder,sess,para,istraining):
        self.placeholder = placeholder
        self.istraining = istraining
        self.batchsize = batchsize
        self.session = sess
        self.model = model
        self.para = para
        self.op_list = []


    def set_feed_dict(self,input):
        scoor,label,coor,graph,index = input

        feed_dict = {
            self.placeholder['isTraining']: self.istraining,
            self.placeholder['batch_size']: self.batchsize,
            self.placeholder['coordinate']: coor[0] if self.para.useSphericalPos else scoor,
        }
        for i in self.para.GN:
            feed_dict[self.placeholder['graph_'+str(i+1)]] = graph[i]
            feed_dict[self.placeholder['poolIndex_' + str(i + 1)]] = index[i]

        return feed_dict

    def __call__(self, input):
        return self.session.run(self.op_list,feed_dict=self.feed_dict)

    def prepare_input(self,input):
        N = self.para.GN
        coor = graph = [0]*N
        index = [0]*(N-1)
        for i in range(1,self.para.GN):
            index[i], coor[i] = utils.farthest_sampling_new(coor[i-1], batch_size=self.batchsize, M=self.para.vertexNum[i],
                                                            k=self.para.poolNum[i],r=self.para.poolRange[i], nodes_n=self.para.vertexNum[i+1])
            graph[i] = utils.middle_graph_generation(coor[i], batch_size=self.batchsize, M=self.para.vertexNum[i],K=self.para.edgeNum[i])


