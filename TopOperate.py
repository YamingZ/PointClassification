import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
import numpy as np
import math
import scipy
import utils
import tf_utils

class TopOperate(object):
    def __init__(self,placeholder,model,para,sess,weight_dict=None,data=None):
        self.model = model
        if data:
            self.combineData(data)
        self.para = para
        self.sess = sess
        self.weightDict = weight_dict
        self.placeholder = placeholder
        self.epoch_count = 0
        self.train_batch_count = 0
        self.eval_batch_count = 0


    def combineData(self,data):
        #objects in data are dict
        train_coor,train_graph,train_label,eval_coor,eval_graph,eval_label = data

        self.train_coor = np.concatenate([value for value in train_coor.values()])
        self.train_label = label_binarize(np.concatenate([value for value in train_label.values()]), classes=[j for j in range(40)])
        self.train_graph = scipy.sparse.vstack(list(train_graph.values())).tocsr()

        self.eval_coor = np.concatenate([value for value in eval_coor.values()])
        self.eval_label = label_binarize(np.concatenate([value for value in eval_label.values()]), classes=[j for j in range(40)])
        self.eval_graph = scipy.sparse.vstack(list(eval_graph.values())).tocsr()


    def trainOneEpoch(self,writer):
        #训练数据集乱序输入
        xTrain, graphTrain, labelTrain = shuffle(self.train_coor, self.train_graph, self.train_label)
        batchSize = self.para.batchSize
        for batchID in range(math.floor(len(labelTrain) / batchSize)): #each batch
            start = batchID * batchSize
            end = start + batchSize
            batchCoor, batchGraph, batchLabel = utils.get_mini_batch(xTrain, graphTrain, labelTrain, start, end)
            # 输入点云数据是否旋转
            if self.para.isRotation:
                batchCoor = utils.rotate_point_cloud(batchCoor)
            #点云数据添加随机抖动
            batchCoor = utils.jitter_point_cloud(batchCoor,sigma=0.008, clip=0.02)
            #点云数据由笛卡尔坐标系转为球坐标系 (x,y,z)->(r,theta,phi)
            batchSCoor = utils.get_Spherical_coordinate(batchCoor)
            batchGraph = batchGraph.todense()
            # 为非均匀数据集加入每种对象类型占比
            if self.para.weighting_scheme == 'uniform':
                batchWeight = utils.uniform_weight(batchLabel)
            elif self.para.weighting_scheme == 'weighted':
                batchWeight = utils.weights_calculation(batchLabel, self.weightDict)
            else:
                print('please enter a valid weighting scheme')
                batchWeight = utils.uniform_weight(batchLabel)

            batchIndexL1, centroid_coordinatesL1 = utils.farthest_sampling_new(batchCoor,
                                                                             M=self.para.clusterNumberL1,
                                                                             k=self.para.nearestNeighborL1,
                                                                             batch_size=batchSize,
                                                                             nodes_n=self.para.pointNumber)

            batchMiddleGraph_1 = utils.middle_graph_generation(centroid_coordinatesL1,
                                                             batch_size=batchSize,
                                                             M=self.para.clusterNumberL1)

            batchIndexL2, centroid_coordinatesL2 = utils.farthest_sampling_new(centroid_coordinatesL1,
                                                                             M=self.para.clusterNumberL2,
                                                                             k=self.para.nearestNeighborL2,
                                                                             batch_size=batchSize,
                                                                             nodes_n=self.para.clusterNumberL1)

            batchMiddleGraph_2 = utils.middle_graph_generation(centroid_coordinatesL2,
                                                             batch_size=batchSize,
                                                             M=self.para.clusterNumberL2)

            feed_dict = {self.placeholder['isTraining']: True,
                         self.placeholder['batch_size']: batchSize,
                         self.placeholder['coordinate']: batchSCoor,
                         self.placeholder['label']: batchLabel,
                         self.placeholder['weights']: batchWeight,
                         self.placeholder['graph_1']: batchGraph,           #for 1st gcn layer graph
                         self.placeholder['batch_index_l1']: batchIndexL1,  #for 1st pooling layer
                         self.placeholder['graph_2']: batchMiddleGraph_1,   #for 2st gcn layer graph
                         self.placeholder['batch_index_l2']: batchIndexL2,  #for 2st pooling layer
                         self.placeholder['graph_3']: batchMiddleGraph_2,   #for 3st gcn layer graph
                         }
            opt,summary = self.sess.run(
                [self.model.opt_op,
                 self.model.summary],
                feed_dict=feed_dict)

            writer.add_summary(summary, self.train_batch_count)
            print("train epoch:{},batch:{}".format(self.epoch_count,batchID))
            self.train_batch_count += 1
        self.epoch_count += 1


    def evaluateOneEpoch(self,writer):
        xTrain, graphTrain, labelTrain = shuffle(self.eval_coor, self.eval_graph, self.eval_label)
        batchSize = self.para.evalBatchSize
        for batchID in range(math.floor(len(labelTrain) / batchSize)): #each batch
            start = batchID * batchSize
            end = start + batchSize
            batchCoor, batchGraph, batchLabel = utils.get_mini_batch(xTrain, graphTrain, labelTrain, start, end)
            if self.para.isRotation == True:
                batchCoor = utils.rotate_point_cloud(batchCoor)
            batchCoor = utils.jitter_point_cloud(batchCoor,sigma=0.008, clip=0.02)
            batchSCoor = utils.get_Spherical_coordinate(batchCoor)  #(x,y,z)->(r,theta,phi)
            batchGraph = batchGraph.todense()
            if self.para.weighting_scheme == 'uniform':
                batchWeight = utils.uniform_weight(batchLabel)
            elif self.para.weighting_scheme == 'weighted':
                batchWeight = utils.weights_calculation(batchLabel, self.weightDict)
            else:
                print('please enter a valid weighting scheme')
                batchWeight = utils.uniform_weight(batchLabel)

            batchIndexL1, centroid_coordinatesL1 = utils.farthest_sampling_new(batchCoor,
                                                                             M=self.para.clusterNumberL1,
                                                                             k=self.para.nearestNeighborL1,
                                                                             batch_size=batchSize,
                                                                             nodes_n=self.para.pointNumber)

            batchMiddleGraph_1 = utils.middle_graph_generation(centroid_coordinatesL1,
                                                             batch_size=batchSize,
                                                             M=self.para.clusterNumberL1)

            batchIndexL2, centroid_coordinatesL2 = utils.farthest_sampling_new(centroid_coordinatesL1,
                                                                             M=self.para.clusterNumberL2,
                                                                             k=self.para.nearestNeighborL2,
                                                                             batch_size=batchSize,
                                                                             nodes_n=self.para.clusterNumberL1)

            batchMiddleGraph_2 = utils.middle_graph_generation(centroid_coordinatesL2,
                                                             batch_size=batchSize,
                                                             M=self.para.clusterNumberL2)

            feed_dict = {self.placeholder['isTraining']: False,
                         self.placeholder['batch_size']: batchSize,
                         self.placeholder['coordinate']: batchSCoor,
                         self.placeholder['label']: batchLabel,
                         self.placeholder['weights']: batchWeight,
                         self.placeholder['graph_1']: batchGraph,           #for 1st gcn layer graph
                         self.placeholder['batch_index_l1']: batchIndexL1,  #for 1st pooling layer
                         self.placeholder['graph_2']: batchMiddleGraph_1,   #for 2st gcn layer graph
                         self.placeholder['batch_index_l2']: batchIndexL2,  #for 2st pooling layer
                         self.placeholder['graph_3']: batchMiddleGraph_2,   #for 3st gcn layer graph
                         }
            acc,loss,summary = self.sess.run(
                [self.model.accuracy,
                 self.model.loss,
                 self.model.summary],
                feed_dict=feed_dict)

            writer.add_summary(summary, self.eval_batch_count)
            print("evaluate epoch:{},batch:{},accuracy:{:.4f},loss:{:.4f}".format(self.epoch_count - 1,batchID,acc,loss))
            self.eval_batch_count += 1

    def predictOneData(self, data):
        coordinate, graph, label = data[0], data[1], data[2]
        batchSize = self.para.testBatchSize
        graph = graph.todense()
        coordinate = utils.get_Spherical_coordinate(coordinate)
        IndexL1, centroid_coordinates_1 = utils.farthest_sampling_new(coordinate,
                                                                    M=self.para.clusterNumberL1,
                                                                    k=self.para.nearestNeighborL1,
                                                                    batch_size=batchSize,
                                                                    nodes_n=self.para.pointNumber)
        MiddleGraph_1 = utils.middle_graph_generation(centroid_coordinates_1, batch_size=1, M=self.para.clusterNumberL1)
        IndexL2, centroid_coordinates_2 = utils.farthest_sampling_new(centroid_coordinates_1,
                                                                    M=self.para.clusterNumberL2,
                                                                    k=self.para.nearestNeighborL2,
                                                                    batch_size=batchSize,
                                                                    nodes_n=self.para.clusterNumberL1)
        MiddleGraph_2 = utils.middle_graph_generation(centroid_coordinates_2, batch_size=1, M=self.para.clusterNumberL2)

        feed_dict = {self.placeholder['isTraining']: False,
                     self.placeholder['batch_size']: batchSize,
                     self.placeholder['coordinate']: coordinate,
                     self.placeholder['graph_1']: graph,  # for 1st gcn layer graph
                     self.placeholder['batch_index_l1']: IndexL1,  # for 1st pooling layer
                     self.placeholder['graph_2']: MiddleGraph_1,  # for 2st gcn layer graph
                     self.placeholder['batch_index_l2']: IndexL2,  # for 2st pooling layer
                     self.placeholder['graph_3']: MiddleGraph_2,  # for 3st gcn layer graph
                    }

        probability = self.sess.run(
            [self.model.probability],
            feed_dict=feed_dict)

        return probability
