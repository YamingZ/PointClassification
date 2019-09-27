import read_data
from HDF5_loader import *
import numpy as np
from utils import *

shape_name=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
            'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar',
            'keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant',
            'radio','range_hood','sink','sofa','stairs','stool','table','tent','toilet',
            'tv_stand','vase','wardrobe','xbox']

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

models = []
#---------------------------------way 1-------------------------------------------------
# for i in range(40):
#     coor = np.array(trainCoor[class_index[i][0]])
#     model = Model(color=[0.0,0.0,0.0],pointsize=5,vertex=coor,visible=True)  #1.0,0.8,0.5
#     models.append(model)

#---------------------------------way 2-------------------------------------------------
for i in range(40):
    coor = np.array(trainCoor[class_index[i][0]])[0:512]
    tree = cKDTree(coor)
    dd, ii = tree.query(coor, k=20)
    A = adjacency(dd, ii)
    B = scipy.sparse.triu(A, k=1)

    lines = []
    coor = coor.tolist()
    for i in range(B.nnz):
        line = coor[B.row[i]] + coor[B.col[i]] + [B.data[i]]
        lines.append(line)

    model = Model(lines=lines, visible=True)  # 1.0,0.8,0.5
    models.append(model)

#---------------------------------way 3-------------------------------------------------
# def pseudo_color(gray):
#     if gray < 0.25:
#         r = 0.0
#         g = 4*gray
#         b = 1.0
#     elif gray < 0.5:
#         r = 0.0
#         g = 1.0
#         b = 2.0 - 4*gray
#     elif gray <0.75:
#         r = 4*gray-2.0
#         g = 1.0
#         b = 0
#     else:
#         r = 1.0
#         g = 4.0 - 4*gray
#         b = 0.0
#     return (r,g,b)
# for i in range(40):
#     coor = np.array([trainCoor[class_index[i][0]]])
#     scoor = get_Spherical_coordinate(coor).squeeze()
#     color = np.array([pseudo_color(c) for c in normalization(scoor[:,0])])
#     # print(color.shape)
#     # color = np.tile(normalization(scoor[:,0:1]),(1,3))
#     rcoor = np.array([[np.sin(coo[1])*np.cos(coo[2]),
#               np.sin(coo[1])*np.sin(coo[2]),
#               np.cos(coo[1])]for coo in scoor])
#     # print(rcoor)
#
#     model = Model(color=color,pointsize=5,vertex=rcoor,visible=True,colorful=True)  #1.0,0.8,0.5
#     models.append(model)


camera = Camera()
view = View3D(models,camera,[1.0,1.0,1.0,0.0],(1024,640),"3Dview")  #0.5,0.6,0.5