import time
import models
import read_data
from parameters import *
from TopOperate import *
import tensorflow as tf
from data import DataSets

def trainOneEpoch(writer, sess, para, placeholders, model, train_dataset, epochCount):
    global TRAINBATCHCOUNT
    batchSize = para.trainBatchSize
    train_iter = train_dataset.iter(batchSize)
    batch_count = 0
    while True:
        try:
            batchSCoor, batchCoor, batchGraph, batchLabel = next(train_iter)
        except StopIteration:
            break

        feed_dict = {placeholders['isTraining']: True,
                     placeholders['coordinate']: batchCoor,
                     placeholders['label']: batchLabel,
                     }
        opt, summary = sess.run(
            [model.opt_op,
             model.summary],
            feed_dict=feed_dict)

        writer.add_summary(summary, TRAINBATCHCOUNT)
        print("train epoch:{},batch:{}".format(epochCount, batch_count))
        batch_count += 1
        TRAINBATCHCOUNT += 1


def evaluateOneEpoch(sess, para, placeholders, model, eval_dataset, epochCount):
    batchSize = para.evalBatchSize
    eval_iter = eval_dataset.iter(batchSize)
    batch_count = 0
    probability_list = []
    label_one_hot_list = []

    while True:
        try:
            batchSCoor, batchCoor, batchGraph, batchLabel = next(eval_iter)
        except StopIteration:
            break

        feed_dict = {placeholders['isTraining']: False,
                     placeholders['coordinate']: batchCoor}

        probability = sess.run(model.probability,feed_dict=feed_dict)

        probability_list.append(probability)
        label_one_hot_list.append(batchLabel)
        print("evaluate epoch:{},batch:{}".format(epochCount, batch_count))
        batch_count += 1

    probabilitys = np.concatenate(probability_list)
    predicts = np.argmax(probabilitys, axis=1)
    label_one_hots = np.concatenate(label_one_hot_list)
    labels = np.argmax(label_one_hots, axis=1)

    confusion_matrix = sklearn.metrics.confusion_matrix(labels, predicts)  # 混淆矩阵
    accuracy = sklearn.metrics.accuracy_score(labels, predicts, normalize=True, sample_weight=None)  # 总准确率
    f1 = sklearn.metrics.f1_score(labels, predicts, average='macro')  # 查准率和查全率的调和平均，1-best，0-worst
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(label_one_hots.ravel(), probabilitys.ravel())
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label_one_hots.ravel(), probabilitys.ravel())  # ROC曲线图
    auc = sklearn.metrics.auc(fpr, tpr)  # AUC

    print("evaluate epoch:{},accuracy:{:.4f},auc:{:.4f}".format(epochCount, accuracy, auc))
    np.savez(para.evalDir + 'eval_epoch_' + str(epochCount) + '.npz', confusion_matrix=confusion_matrix,
             accuracy=accuracy, precision=precision, recall=recall, f1=f1, fpr=fpr, tpr=tpr, auc=auc)

# ===============================Hyper parameters========================
para = Parameters()
para.info()
para.log()
# ============================Define placeholders==========================
placeholders = {
    'isTraining': tf.placeholder(tf.bool,name='is_training'),
    'coordinate': tf.placeholder(tf.float32, [None, para.pointNumber, para.input_data_dim], name='coordinate'),
    'label': tf.placeholder(tf.float32, [None, para.outputClassN], name='label'),
}
# ================================Load data===============================
inputTrain, trainLabel, inputTest, testLabel = read_data.load_data(para.pointNumber, para.samplingType, para.dataDir)
scaledLaplacianTrain, scaledLaplacianTest = read_data.prepareData(inputTrain, inputTest, para.neighborNumber, para.pointNumber,para.dataDir)
# ================================Create model===============================
model = models.PointNet(para,placeholders,logging=True)
# =============================Initialize session============================
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
# ===============================Train model ================================
TRAINBATCHCOUNT = 0
for epoch in range(para.max_epoch):
    train_dataset = DataSets((inputTrain, scaledLaplacianTrain, trainLabel),rotate=para.isRotationTrain)
    train_start_time = time.time()
    trainOneEpoch(train_writer,sess,para,placeholders,model,train_dataset,epoch)  #writer, sess, para, placeholders, model, train_dataset, trainBatchCount, epochCount
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    print("train epoch {} cost time is {} second".format(epoch,train_time))
    if para.EvalCycle:
        if epoch % para.EvalCycle == 0:  #evaluate model after every two training epoch
            eval_dataset = DataSets((inputTest, scaledLaplacianTest, testLabel),rotate=para.isRotationEval)
            eval_start_time = time.time()
            evaluateOneEpoch(sess,para,placeholders,model,eval_dataset,epoch) #sess, para, placeholders, model, eval_dataset, epochCount
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            print("eval epoch {} cost time is {} second".format(epoch, eval_time))
model.save(para.ckptDir,sess)



