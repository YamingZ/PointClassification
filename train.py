import time
import read_data
from parameters import *
from TopOperate import *
import tensorflow as tf
from data import DataSets
from RPGCN import *

# ===============================Hyper parameters========================
para = Parameters(roadparam=False)
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
    'poolIndex_1': tf.placeholder(tf.int32, [None, para.vertexNumG2 * para.poolNumG1], name='poolIndex1'),
    'poolIndex_2': tf.placeholder(tf.int32, [None, para.vertexNumG3 * para.poolNumG2], name='poolIndex2'),
    'poolIndex_3': tf.placeholder(tf.int32, [None, para.vertexNumG4 * para.poolNumG3], name='poolIndex3')
    # 'lr': tf.placeholder(tf.float32, name='lr'),
}
# ================================Load data===============================
inputTrain, trainLabel, inputTest, testLabel = read_data.load_data(para.vertexNumG1, para.samplingType, para.dataDir)
# layer_1: (1)graph generate
scaledLaplacianTrain, scaledLaplacianTest = read_data.prepareData(inputTrain, inputTest, para.edgeNumG1, para.vertexNumG1, para.dataDir)
train_weight_dict = utils.train_weight_dict(trainLabel, para)
eval_weight_dict = utils.eval_weight_dict(testLabel)
# ================================Create model===============================
model = RPGCN(para,placeholders,logging=True)
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
