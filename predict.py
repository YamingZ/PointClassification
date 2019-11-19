from RPGCN import *
import read_data
from parameters import *
from TopOperate import *
from utils import *
import tensorflow as tf

# ===============================Hyper parameters========================
para = Parameters(roadparam=True)
para.info()
para.save()
# ============================Define placeholders==========================
placeholders = {
    'isTraining': tf.placeholder(tf.bool,name='is_training'),
    'batch_size': tf.placeholder(tf.int32,name='batch_size'),
    'coordinate': tf.placeholder(tf.float32, [None, para.vertexNumG1, para.input_data_dim], name='coordinate'),
    'label': tf.placeholder(tf.float32, [None, para.outputClassN], name='label'),
    'weights': tf.placeholder(tf.float32, [None], name='weights'),
    'graph_1': tf.placeholder(tf.float32, [None, para.vertexNumG1 * para.vertexNumG1], name='graph1'),
    'graph_2': tf.placeholder(tf.float32, [None, para.vertexNumG2 * para.vertexNumG2], name='graph2'),
    'graph_3': tf.placeholder(tf.float32, [None, para.vertexNumG3 * para.vertexNumG3], name='graph3'),
    'graph_4': tf.placeholder(tf.float32, [None, para.vertexNumG4 * para.vertexNumG4], name='graph4'),
    'poolIndex_1': tf.placeholder(tf.int32, [None, para.vertexNumG2 * para.poolNumG1], name='poolIndex1'),
    'poolIndex_2': tf.placeholder(tf.int32, [None, para.vertexNumG3 * para.poolNumG2], name='poolIndex2'),
    'poolIndex_3': tf.placeholder(tf.int32, [None, para.vertexNumG4 * para.poolNumG3], name='poolIndex3')
}

shape_name=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
            'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar',
            'keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant',
            'radio','range_hood','sink','sofa','stairs','stool','table','tent','toilet',
            'tv_stand','vase','wardrobe','xbox']

# ================================Load data===============================
data,label = load_h5(para.dataDir+"modelnet40/modelnet40_ply_hdf5_2048/ply_data_test0.h5")

coor = np.array(data)[1000,0:1024,:]
coor = np.expand_dims(coor,axis=0)
label = np.array(label)[1000][0]
coor_rotate = rotate_point_cloud(coor)
coor_jitter = jitter_point_cloud(coor_rotate)

scaledLaplacian = read_data.get_scaledLaplacian(coor_jitter,para.edgeNumG1, para.vertexNumG1)
Data = (coor_jitter,scaledLaplacian,label)


# ================================Create model===============================
model = RPGCN(para,placeholders,logging=True)
# =============================Initialize session=============================
sess = tf.Session()
# ==============================Init variables===============================
model.load(para.ckptDir,sess)
# =============================Graph Visualizing=============================

# ==============================Init variables===============================

# ===============================test data ================================
top_op = TopOperate(placeholders,model,para,sess)
probability = np.squeeze(top_op.predictOneData(Data))

sort_probability = probability[np.argsort(-probability)]
print(sort_probability)
shape_name = np.array(shape_name)
sort_shape_name = shape_name[np.argsort(-probability)]
print(sort_shape_name)

print('predict: {} '.format(np.argmax(probability))+ shape_name[int(np.argmax(probability))])
print('groundtruth: {} '.format(label) + shape_name[int(label)])