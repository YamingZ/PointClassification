import time
import models
import read_data
from parameters import *
from TopOperate import *
import tensorflow as tf

# ===============================Hyper parameters========================
para = Parameters()
para.info()
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
# ================================Load data===============================
inputTrain, trainLabel, inputTest, testLabel = read_data.load_data(para.pointNumber, para.samplingType, para.dataDir)
scaledLaplacianTrain, scaledLaplacianTest = read_data.prepareData(inputTrain, inputTest, para.neighborNumber, para.pointNumber,para.dataDir)
data = (inputTrain,scaledLaplacianTrain,trainLabel,inputTest,scaledLaplacianTest,testLabel)    #train data & test data
weight_dict = utils.weight_dict_fc(trainLabel, para)
# ================================Create model===============================

def get_slice(data, i, parts):
    shape = tf.shape(data)  #[28,512,3]
    batch_size = shape[:1]
    input_shape = shape[1:]
    step = batch_size // parts
    if i == parts - 1:
        size = batch_size - step * i
    else:
        size = step
    size = tf.concat([size, input_shape], axis=0)
    stride = tf.concat([step, input_shape * 0], axis=0)
    start = stride * i
    return tf.slice(data, start, size)

#在同一模型的不同实例中共享相同变量
#在同一模型中不共享重复层的变量
def mutil_gpu_model(gpu_nums):
    # with tf.device('/cpu:0'):
        tower_grads = []
        for i in range(gpu_nums):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    model = models.GPN(para, placeholders, logging=True)
                    model.dataInput(
                        get_slice(placeholders['coordinate'], i, gpu_nums),
                        get_slice(placeholders['label'], i, gpu_nums),
                        get_slice(placeholders['weights'], i, gpu_nums),
                        get_slice(placeholders['graph_1'], i, gpu_nums),
                        get_slice(placeholders['graph_2'], i, gpu_nums),
                        get_slice(placeholders['graph_3'], i, gpu_nums),
                        get_slice(placeholders['batch_index_l1'], i, gpu_nums),
                        get_slice(placeholders['batch_index_l2'], i, gpu_nums),
                    )
                    tower_loss = model.tower_loss(scope)
                    # tf.get_variable的命名空间
                    tf.get_variable_scope().reuse_variables()
                    # 使用当前gpu计算所有变量的梯度
                    grads = model.optimizer.compute_gradients(tower_loss)
                    tower_grads.append(grads)
        # 计算变量的平均梯度
        grads = tf_utils.average_gradients(tower_grads)
        # 使用平均梯度更新参数
        apply_gradient_op = model.optimizer.apply_gradients(grads)
        return apply_gradient_op

# =============================Initialize session=============================
sess = tf.Session()
# ==============================Init variables===============================
if para.restoreModel:
    model.load(sess)
else:
    sess.run(tf.global_variables_initializer())
# =============================Graph Visualizing=============================
TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
merged_summary = tf.summary.merge_all()
# evaluation log
eval_log_dir = "tensorboard/eval/"+TIMESTAMP
eval_writer = tf.summary.FileWriter(eval_log_dir)
# train log
train_log_dir = "tensorboard/train/"+TIMESTAMP
train_writer = tf.summary.FileWriter(train_log_dir)
train_writer.add_graph(sess.graph)
# ===============================Train model ================================
train_coor = np.concatenate([value for value in inputTrain.values()])
train_label = label_binarize(np.concatenate([value for value in trainLabel.values()]), classes=[j for j in range(40)])
train_graph = scipy.sparse.vstack(list(scaledLaplacianTrain.values())).tocsr()
xTrain, graphTrain, labelTrain = shuffle(train_coor, train_graph, train_label)
batchSize = para.batchSize
for batchID in range(math.floor(len(labelTrain) / batchSize)):  # each batch
    start = batchID * batchSize
    end = start + batchSize
    batchCoor, batchGraph, batchLabel = utils.get_mini_batch(xTrain, graphTrain, labelTrain, start, end)
    # 输入点云数据是否旋转
    if para.isRotation:
        batchCoor = utils.rotate_point_cloud(batchCoor)
    # 点云数据添加随机抖动
    batchCoor = utils.jitter_point_cloud(batchCoor, sigma=0.008, clip=0.02)
    # 点云数据由笛卡尔坐标系转为球坐标系 (x,y,z)->(r,theta,phi)
    batchSCoor = utils.get_Spherical_coordinate(batchCoor)
    batchGraph = batchGraph.todense()
    # 为非均匀数据集加入每种对象类型占比
    if para.weighting_scheme == 'uniform':
        batchWeight = utils.uniform_weight(batchLabel)
    elif para.weighting_scheme == 'weighted':
        batchWeight = utils.weights_calculation(batchLabel, weight_dict)
    else:
        print('please enter a valid weighting scheme')
        batchWeight = utils.uniform_weight(batchLabel)

    batchIndexL1, centroid_coordinatesL1 = utils.farthest_sampling_new(batchCoor,
                                                                       M=para.clusterNumberL1,
                                                                       k=para.nearestNeighborL1,
                                                                       batch_size=batchSize,
                                                                       nodes_n=para.pointNumber)

    batchMiddleGraph_1 = utils.middle_graph_generation(centroid_coordinatesL1,
                                                       batch_size=batchSize,
                                                       M=para.clusterNumberL1)

    batchIndexL2, centroid_coordinatesL2 = utils.farthest_sampling_new(centroid_coordinatesL1,
                                                                       M=para.clusterNumberL2,
                                                                       k=para.nearestNeighborL2,
                                                                       batch_size=batchSize,
                                                                       nodes_n=para.clusterNumberL1)

    batchMiddleGraph_2 = utils.middle_graph_generation(centroid_coordinatesL2,
                                                       batch_size=batchSize,
                                                       M=para.clusterNumberL2)

    feed_dict = {placeholders['isTraining']: True,
                 placeholders['batch_size']: batchSize,
                 placeholders['coordinate']: batchSCoor,
                 placeholders['label']: batchLabel,
                 placeholders['weights']: batchWeight,
                 placeholders['graph_1']: batchGraph,  # for 1st gcn layer graph
                 placeholders['batch_index_l1']: batchIndexL1,  # for 1st pooling layer
                 placeholders['graph_2']: batchMiddleGraph_1,  # for 2st gcn layer graph
                 placeholders['batch_index_l2']: batchIndexL2,  # for 2st pooling layer
                 placeholders['graph_3']: batchMiddleGraph_2,  # for 3st gcn layer graph
                 }

    train_op = mutil_gpu_model(para.GpuNums)

    opt, summary = sess.run(
        [train_op,
         model.summary],
        feed_dict=feed_dict)
