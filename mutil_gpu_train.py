import time
import models
import read_data
from parameters import *
from TopOperate import *
import tensorflow as tf

# ===============================Hyper parameters========================
para = Parameters()
para.info()
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

with tf.Graph().as_default(),tf.device('/cpu:0'):
    # define an optimizer in cpu
    global_step = tf.get_variable('global_step',dtype=tf.int32,initializer=tf.constant(0),trainable=False)
    lr = tf.train.exponential_decay(para.learningRate,      #初始学习率
                                    global_step,            #Variable，每batch加一
                                    para.lr_decay_steps,    #global_step/decay_steps得到decay_rate的幂指数
                                    0.96,                   #学习率衰减系数
                                    staircase=False)        #若True ，则学习率衰减呈离散间隔
    lr = tf.maximum(lr, para.minimum_lr)
    tf.summary.scalar("learning_rate", lr)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # ============================Define placeholders==========================
    placeholders = {
        'isTraining'    : tf.placeholder(tf.bool, name='is_training'),
        'batch_size'    : tf.placeholder(tf.int32, name='batch_size'),
        'coordinate'    : tf.placeholder(tf.float32, [None, para.pointNumber, para.input_data_dim], name='coordinate'),
        'label'         : tf.placeholder(tf.float32, [None, para.outputClassN], name='label'),
        'weights'       : tf.placeholder(tf.float32, [None], name='weights'),
        'graph_1'       : tf.placeholder(tf.float32, [None, para.pointNumber * para.pointNumber], name='graph1'),
        'graph_2'       : tf.placeholder(tf.float32, [None, para.clusterNumberL1 * para.clusterNumberL1], name='graph2'),
        'graph_3'       : tf.placeholder(tf.float32, [None, para.clusterNumberL2 * para.clusterNumberL2], name='graph3'),
        'batch_index_l1': tf.placeholder(tf.int32, [None, para.clusterNumberL1 * para.nearestNeighborL1],name='batch_index_l1'),
        'batch_index_l2': tf.placeholder(tf.int32, [None, para.clusterNumberL2 * para.nearestNeighborL2],name='batch_index_l2')
    }
    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in range(para.GpuNums):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('GPU_%d' % i) as scope:
                #input data for each model tower
                model = models.GPN(para, placeholders, logging=True)
                model.dataInput(
                    get_slice(placeholders['coordinate'], i, para.GpuNums),
                    get_slice(placeholders['label'], i, para.GpuNums),
                    get_slice(placeholders['weights'], i, para.GpuNums),
                    get_slice(placeholders['graph_1'], i, para.GpuNums),
                    get_slice(placeholders['graph_2'], i, para.GpuNums),
                    get_slice(placeholders['graph_3'], i, para.GpuNums),
                    get_slice(placeholders['batch_index_l1'], i, para.GpuNums),
                    get_slice(placeholders['batch_index_l2'], i, para.GpuNums)
                )
                # build model inference for each model tower
                loss = model.tower_loss(scope)
                # 在所有设备中复用参数
                tf.get_variable_scope().reuse_variables()
                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                # 使用当前gpu计算所有变量的梯度
                grads = optimizer.compute_gradients(loss)
                # Keep track of the gradients across all towers
                tower_grads.append(grads)

    # 计算变量的平均梯度
    grads = tf_utils.average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # 使用平均梯度更新参数
    apply_gradient_op = optimizer.apply_gradients(grads)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.9, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    sess.run(init)

    TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    train_log_dir = "tensorboard/train/" + TIMESTAMP
    summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

# ================================Load data===============================
    inputTrain, trainLabel, inputTest, testLabel = read_data.load_data(para.pointNumber, para.samplingType,
                                                                       para.dataDir)
    scaledLaplacianTrain, scaledLaplacianTest = read_data.prepareData(inputTrain, inputTest, para.neighborNumber,
                                                                      para.pointNumber, para.dataDir)
    weight_dict = utils.weight_dict_fc(trainLabel, para)
    train_coor = np.concatenate([value for value in inputTrain.values()])
    train_label = label_binarize(np.concatenate([value for value in trainLabel.values()]), classes=[j for j in range(40)])
    train_graph = scipy.sparse.vstack(list(scaledLaplacianTrain.values())).tocsr()
# ===============================Train model ================================
    # one epoch
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

        opt, summary = sess.run(
            [train_op,
             summary_op],
            feed_dict=feed_dict)

