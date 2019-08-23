import models
import read_data
from parameters import *
from TopOperate import *
from utils import *
import tensorflow as tf

# ===============================Hyper parameters========================
para = Parameters()
# ============================Define placeholders==========================
placeholders = {
    'isTraining': tf.placeholder(tf.bool,name='is_training'),
    'batch_size': tf.placeholder(tf.int32,name='batch_size'),
    'coordinate': tf.placeholder(tf.float32, [None, para.pointNumber, para.input_data_dim], name='coordinate'),
    'label': tf.placeholder(tf.float32, [None, para.outputClassN], name='label'),
    'weights': tf.placeholder(tf.float32, [None], name='weights'),
    'graph_1': tf.placeholder(tf.float32, [None, para.pointNumber * para.pointNumber], name='graph1'),
    'graph_2': tf.placeholder(tf.float32, [None, para.clusterNumberL1 * para.clusterNumberL1], name='graph2'),
    'graph_3': tf.placeholder(tf.float32, [None, para.clusterNumberL2 * para.clusterNumberL2], name='graph3'),
    'batch_index_l1': tf.placeholder(tf.int32, [None, para.clusterNumberL1 * para.nearestNeighborL1],name='batch_index_l1'),
    'batch_index_l2': tf.placeholder(tf.int32, [None, para.clusterNumberL2 * para.nearestNeighborL2],name='batch_index_l2')
}

shape_name=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
            'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar',
            'keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant',
            'radio','range_hood','sink','sofa','stairs','stool','table','tent','toilet',
            'tv_stand','vase','wardrobe','xbox']

# ================================Load data===============================
data,label = load_h5(para.dataDir+"modelnet/modelnet40_ply_hdf5_2048/ply_data_test0.h5")

coor = np.array(data)[66,0:512,:]
coor = np.expand_dims(coor,axis=0)
label = np.array(label)[66][0]
coor_rotate = rotate_point_cloud(coor)
coor_jitter = jitter_point_cloud(coor_rotate)

scaledLaplacian = read_data.get_scaledLaplacian(coor_jitter,para.neighborNumber,para.pointNumber)
Data = (coor_jitter,scaledLaplacian,label)


# # ================================Create model===============================
model = models.GPN(para,placeholders,logging=False)
# # =============================Initialize session=============================
sess = tf.Session()
model.load(sess)
# =============================Graph Visualizing=============================

# ==============================Init variables===============================

# ===============================test data ================================
top_op = TopOperate(placeholders,model,para,sess)
probability = top_op.predictOneData(Data)

print(probability)
print('predict: {} '.format(np.argmax(probability))+ shape_name[int(np.argmax(probability))])
print('groundtruth: {} '.format(label) + shape_name[int(label)])