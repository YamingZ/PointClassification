from HDF5_loader import *
from parameters import *
import numpy as np
import read_data
import utils

para = Parameters()


inputTrain, trainLabel, inputTest, testLabel = read_data.load_data(2048, 'farthest_sampling','/home/ym/PycharmProjects/Fundamentals_of_Python/PointClassfication/data/')
trainLabel = np.concatenate([value for value in trainLabel.values()]).tolist()
testLabel = np.concatenate([value for value in testLabel.values()]).tolist()
trainCoor = np.concatenate([value for value in inputTrain.values()])
testCoor = np.concatenate([value for value in inputTest.values()])

class_index = [[] for i in range(40)]
for i in range(40):
    for n, label in enumerate(trainLabel):
        if label==i:
            class_index[i].append(n)

coor = np.array(trainCoor[class_index[0][0]])
coor = np.expand_dims(coor, axis=0)
print(coor.shape)


IndexL1, centroid_coordinates_1 = utils.farthest_sampling_new(coor,
                                                              M=512,
                                                              k=4,
                                                              batch_size=1,
                                                              nodes_n=2048)
print(centroid_coordinates_1.shape)

IndexL2, centroid_coordinates_2 = utils.farthest_sampling_new(centroid_coordinates_1,
                                                              M=128,
                                                              k=4,
                                                              batch_size=1,
                                                              nodes_n=512)
print(centroid_coordinates_2.shape)


model_1 = Model(vertex=coor.squeeze(), color=[0.2, 0.7, 0.1], pointsize=8)
model_2 = Model(vertex=centroid_coordinates_1.reshape([-1,3]), color=[0.0, 0.0, 1.0], pointsize=9)
model_3 = Model(vertex=centroid_coordinates_2.reshape([-1,3]), color=[1.0, 0.0, 0.0], pointsize=10)

camera = Camera()
view = View3D([model_3,model_2,model_1], camera,[1.0,1.0,1.0,0.0], (1024, 640), "3Dview")   #[0.5,0.6,0.5,0.0]