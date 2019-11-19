import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
import numpy as np
import math
import scipy
import utils
import sklearn.metrics
import tf_utils


class TopOperate(object):
    def __init__(self,placeholder,model,para,sess):
        self.model = model
        self.para = para
        self.sess = sess
        self.placeholder = placeholder
        self.epoch_count = 0
        self.train_batch_count = 0
        self.eval_batch_count = 0


    def trainOneEpoch(self,writer,train_dataset,weight_dict):
        batchSize = self.para.trainBatchSize
        train_iter = train_dataset.iter(batchSize)
        batch_count = 0
        while True:
            try:
                SCoor_1, Coor_1, Graph_1, batchLabel= next(train_iter)
            except StopIteration:
                break
            # 为非均匀数据集加入每种对象类型占比
            if self.para.weighting_scheme == 'uniform':
                batchWeight = utils.uniform_weight(batchLabel)
            elif self.para.weighting_scheme == 'weighted':
                batchWeight = utils.weights_calculation(batchLabel, weight_dict)
            else:
                print('please enter a valid weighting scheme')
                batchWeight = utils.uniform_weight(batchLabel)

            # layer_1: (2)down sampling and pooling
            IndexL1, Coor_2 = utils.farthest_sampling_new(Coor_1,M=self.para.vertexNumG2,k=self.para.poolNumG1,
                                                          r=self.para.poolRangeG1,batch_size=batchSize,nodes_n=self.para.vertexNumG1)

            # layer_2: (1)graph generate (2)down sampling and pooling
            Graph_2 = utils.middle_graph_generation(Coor_2,batch_size=batchSize,M=self.para.vertexNumG2,K=self.para.edgeNumG2)
            IndexL2, Coor_3 = utils.farthest_sampling_new(Coor_2,M=self.para.vertexNumG3,k=self.para.poolNumG2,
                                                          r=self.para.poolRangeG2,batch_size=batchSize,nodes_n=self.para.vertexNumG2)

            # layer_3: (1)graph generate (2)down sampling and pooling
            Graph_3 = utils.middle_graph_generation(Coor_3,batch_size=batchSize,M=self.para.vertexNumG3,K=self.para.edgeNumG3)
            IndexL3, Coor_4 = utils.farthest_sampling_new(Coor_3,M=self.para.vertexNumG4,k=self.para.poolNumG3,
                                                          r=self.para.poolRangeG3,batch_size=batchSize,nodes_n=self.para.vertexNumG3)

            feed_dict = {self.placeholder['isTraining']: True,
                         self.placeholder['batch_size']: batchSize,
                         self.placeholder['coordinate']: SCoor_1 if self.para.useSphericalPos else Coor_1,
                         self.placeholder['label']: batchLabel,
                         self.placeholder['weights']: batchWeight,
                         self.placeholder['graph_1']: Graph_1,       #for 1st gcn layer graph
                         self.placeholder['poolIndex_1']: IndexL1,   #for 1st pooling layer
                         self.placeholder['graph_2']: Graph_2,    #for 2st gcn layer graph
                         self.placeholder['poolIndex_2']: IndexL2,   #for 2st pooling layer
                         self.placeholder['graph_3']: Graph_3,    #for 3st gcn layer graph
                         self.placeholder['poolIndex_3']: IndexL3,   #for 3st pooling layer
                         }
            opt,summary = self.sess.run(
                [self.model.opt_op,
                 self.model.summary],
                feed_dict=feed_dict)

            writer.add_summary(summary, self.train_batch_count)
            print("train epoch:{},batch:{}".format(self.epoch_count,batch_count))
            self.train_batch_count += 1
            batch_count+=1
        self.epoch_count += 1


    def evaluateOneEpoch(self,eval_dataset,weight_dict):
        batchSize = self.para.evalBatchSize
        eval_iter = eval_dataset.iter(batchSize)
        batch_count = 0
        probability_list = []
        label_one_hot_list = []
        batchWeight_list = []
        while True:
            try:
                SCoor_1, Coor_1, Graph_1, batchLabel = next(eval_iter)
            except StopIteration:
                break
            # 为非均匀数据集加入每种对象类型占比
            if self.para.weighting_scheme == 'uniform':
                batchWeight = utils.uniform_weight(batchLabel)
            elif self.para.weighting_scheme == 'weighted':
                batchWeight = utils.weights_calculation(batchLabel, weight_dict)
            else:
                print('please enter a valid weighting scheme')
                batchWeight = utils.uniform_weight(batchLabel)

            # layer_1: (2)down sampling and pooling
            IndexL1, Coor_2 = utils.farthest_sampling_new(Coor_1, M=self.para.vertexNumG2, k=self.para.poolNumG1,
                                                          r=self.para.poolRangeG1, batch_size=batchSize, nodes_n=self.para.vertexNumG1)

            # layer_2: (1)graph generate (2)down sampling and pooling
            Graph_2 = utils.middle_graph_generation(Coor_2, batch_size=batchSize, M=self.para.vertexNumG2,
                                                    K=self.para.edgeNumG2)
            IndexL2, Coor_3 = utils.farthest_sampling_new(Coor_2, M=self.para.vertexNumG3, k=self.para.poolNumG2,
                                                          r=self.para.poolRangeG2, batch_size=batchSize, nodes_n=self.para.vertexNumG2)

            # layer_3: (1)graph generate (2)down sampling and pooling
            Graph_3 = utils.middle_graph_generation(Coor_3, batch_size=batchSize, M=self.para.vertexNumG3,
                                                    K=self.para.edgeNumG3)
            IndexL3, Coor_4 = utils.farthest_sampling_new(Coor_3, M=self.para.vertexNumG4, k=self.para.poolNumG3,
                                                          r=self.para.poolRangeG3, batch_size=batchSize, nodes_n=self.para.vertexNumG3)

            feed_dict = {self.placeholder['isTraining']: False,
                         self.placeholder['batch_size']: batchSize,
                         self.placeholder['coordinate']: SCoor_1 if self.para.useSphericalPos else Coor_1,
                         self.placeholder['graph_1']: Graph_1,  # for 1st gcn layer graph
                         self.placeholder['poolIndex_1']: IndexL1,  # for 1st pooling layer
                         self.placeholder['graph_2']: Graph_2,  # for 2st gcn layer graph
                         self.placeholder['poolIndex_2']: IndexL2,  # for 2st pooling layer
                         self.placeholder['graph_3']: Graph_3,  # for 3st gcn layer graph
                         self.placeholder['poolIndex_3']: IndexL3,  # for 3st pooling layer
                         }

            probability = self.sess.run(
                self.model.probability,
                feed_dict=feed_dict)

            batchWeight_list.append(batchWeight)
            probability_list.append(probability)
            label_one_hot_list.append(batchLabel)
            print("evaluate epoch:{},batch:{}".format(self.epoch_count-1,batch_count))
            batch_count += 1

        batchWeights = np.concatenate(batchWeight_list)
        probabilitys = np.concatenate(probability_list)
        predicts = np.argmax(probabilitys,axis=1)
        label_one_hots = np.concatenate(label_one_hot_list)
        labels = np.argmax(label_one_hots,axis=1)

        confusion_matrix = sklearn.metrics.confusion_matrix(labels,predicts)                            #混淆矩阵
        accuracy = sklearn.metrics.accuracy_score(labels,predicts, normalize=True, sample_weight=batchWeights)  #总准确率
        # precision = sklearn.metrics.precision_score(labels,predicts,average ='macro')                      #查准率 weighted
        # recall = sklearn.metrics.recall_score(labels, predicts, average='macro')                           #查全率
        f1 = sklearn.metrics.f1_score(labels, predicts, average='macro')                                   #查准率和查全率的调和平均，1-best，0-worst
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(label_one_hots.ravel(), probabilitys.ravel())
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(label_one_hots.ravel(), probabilitys.ravel())  #ROC曲线图
        auc = sklearn.metrics.auc(fpr, tpr)                                                             #AUC

        print("evaluate epoch:{},accuracy:{:.4f},auc:{:.4f}".format(self.epoch_count - 1,accuracy,auc))
        np.savez(self.para.evalDir+'eval_epoch_'+str(self.epoch_count-1)+'.npz',confusion_matrix=confusion_matrix,accuracy=accuracy,precision=precision,recall=recall,f1=f1,fpr=fpr,tpr=tpr,auc=auc)


    def predictOneData(self, data):
        Coor_1, graph, label = data[0], data[1], data[2]
        batchSize = self.para.testBatchSize
        Graph_1 = graph.todense()
        SCoor_1 = utils.get_Spherical_coordinate(Coor_1)

        # layer_1: (2)down sampling and pooling
        IndexL1, Coor_2 = utils.farthest_sampling_new(Coor_1, M=self.para.vertexNumG2, k=self.para.poolNumG1,
                                                      r=self.para.poolRangeG1, batch_size=batchSize,
                                                      nodes_n=self.para.vertexNumG1)

        # layer_2: (1)graph generate (2)down sampling and pooling
        Graph_2 = utils.middle_graph_generation(Coor_2, batch_size=batchSize, M=self.para.vertexNumG2,
                                                K=self.para.edgeNumG2)
        IndexL2, Coor_3 = utils.farthest_sampling_new(Coor_2, M=self.para.vertexNumG3, k=self.para.poolNumG2,
                                                      r=self.para.poolRangeG2, batch_size=batchSize,
                                                      nodes_n=self.para.vertexNumG2)

        # layer_3: (1)graph generate (2)down sampling and pooling
        Graph_3 = utils.middle_graph_generation(Coor_3, batch_size=batchSize, M=self.para.vertexNumG3,
                                                K=self.para.edgeNumG3)
        IndexL3, Coor_4 = utils.farthest_sampling_new(Coor_3, M=self.para.vertexNumG4, k=self.para.poolNumG3,
                                                      r=self.para.poolRangeG3, batch_size=batchSize,
                                                      nodes_n=self.para.vertexNumG3)

        feed_dict = {self.placeholder['isTraining']: False,
                     self.placeholder['batch_size']: batchSize,
                     self.placeholder['coordinate']: SCoor_1 if self.para.useSphericalPos else Coor_1,
                     self.placeholder['graph_1']: Graph_1,  # for 1st gcn layer graph
                     self.placeholder['poolIndex_1']: IndexL1,  # for 1st pooling layer
                     self.placeholder['graph_2']: Graph_2,  # for 2st gcn layer graph
                     self.placeholder['poolIndex_2']: IndexL2,  # for 2st pooling layer
                     self.placeholder['graph_3']: Graph_3,  # for 3st gcn layer graph
                     self.placeholder['poolIndex_3']: IndexL3,  # for 3st pooling layer
                     }

        probability = self.sess.run(
            [self.model.probability],
            feed_dict=feed_dict)

        return probability
