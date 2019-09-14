import time
import models
import read_data
from parameters import *
from TopOperate import *
import tensorflow as tf
from data import DataSets

# ===============================Hyper parameters========================
para = Parameters()
para.info()
para.log()
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
    # 'graph_4': tf.placeholder(tf.float32, [None, para.clusterNumberL3 * para.clusterNumberL3], name='graph4'),
    'batch_index_l1': tf.placeholder(tf.int32, [None, para.clusterNumberL1 * para.poolingRange], name='batch_index_l1'),
    'batch_index_l2': tf.placeholder(tf.int32, [None, para.clusterNumberL2 * para.poolingRangeL1],name='batch_index_l2'),
    'batch_index_l3': tf.placeholder(tf.int32, [None, para.clusterNumberL3 * para.poolingRangeL2], name='batch_index_l3')
    # 'lr': tf.placeholder(tf.float32, name='lr'),
}
# ================================Load data===============================
inputTrain, trainLabel, inputTest, testLabel = read_data.load_data(para.pointNumber, para.samplingType, para.dataDir)
scaledLaplacianTrain, scaledLaplacianTest = read_data.prepareData(inputTrain, inputTest, para.neighborNumber, para.pointNumber,para.dataDir)
train_weight_dict = utils.train_weight_dict(trainLabel, para)
eval_weight_dict = utils.eval_weight_dict(testLabel)
# ================================Create model===============================
model = models.GPN(para,placeholders,logging=True)
# =============================Initialize session=============================
sess = tf.Session()
# ==============================Init variables===============================
if para.restoreModel:
    model.load(para.ckptDir,sess)
else:
    sess.run(tf.global_variables_initializer())
# =============================Graph Visualizing=============================
TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
merged_summary = tf.summary.merge_all()
# train log
train_log_dir = "tensorboard/train/"+TIMESTAMP
train_writer = tf.summary.FileWriter(train_log_dir)
train_writer.add_graph(sess.graph)
# evaluation log
# eval_log_dir = "tensorboard/eval/"+TIMESTAMP
# eval_writer = tf.summary.FileWriter(eval_log_dir)
# ===============================Train model ================================
top_op = TopOperate(placeholders,model,para,sess)
for epoch in range(para.max_epoch):
    train_dataset = DataSets((inputTrain, scaledLaplacianTrain, trainLabel),rotate=para.isRotationTrain)
    train_start_time = time.time()
    top_op.trainOneEpoch(train_writer,train_dataset,train_weight_dict)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    print("train epoch {} cost time is {} second".format(epoch,train_time))
    if para.EvalCycle:
        if epoch % para.EvalCycle == 0:  #evaluate model after every two training epoch
            eval_dataset = DataSets((inputTest, scaledLaplacianTest, testLabel),rotate=para.isRotationEval)
            eval_start_time = time.time()
            top_op.evaluateOneEpoch(eval_dataset,eval_weight_dict)
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            print("eval epoch {} cost time is {} second".format(epoch, eval_time))
model.save(para.ckptDir,sess)
