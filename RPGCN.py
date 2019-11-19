from models import *

class RPGCN(Model):
    def __init__(self,para,placeholders,**kwargs):
        super(RPGCN,self).__init__(**kwargs)
        self.para = para
        self.is_training = placeholders['isTraining']
        self.batch_size = placeholders['batch_size']
        # signal training device
        self.inputs = placeholders['coordinate']
        self.other_inputs = placeholders
        self.build()

    def _loss(self):
        # with tf.name_scope('MatLoss'):
        #     self.transform = self.layers[0].transform
        #     K = self.transform.get_shape()[1].value
        #     mat_diff = tf.matmul(self.transform, tf.transpose(self.transform, perm=[0, 2, 1]))
        #     mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
        #     mat_diff_loss = tf.nn.l2_loss(mat_diff)
        # tf.summary.scalar('MatLoss', mat_diff_loss)
        with tf.name_scope('L2_loss'):
            vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'weights' in v.name]) * self.para.l2_rate
        tf.summary.scalar('L2_loss', l2_loss)
        with tf.name_scope('Loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs, labels=self.other_inputs['label'])
            loss = tf.multiply(loss, self.other_inputs['weights'])
            loss = tf.reduce_mean(loss)
            self.loss = loss  + l2_loss #+ mat_diff_loss
        tf.summary.scalar("loss", self.loss)
        tf.add_to_collection('losses', self.loss)

    def _accuracy(self):
        self.probability = tf.nn.softmax(self.outputs)
        self.predictLabels = tf.argmax(self.probability, axis=1) #softmax can be delete
        correct_prediction = tf.equal(self.predictLabels, tf.argmax(self.other_inputs['label'], axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("acc", self.accuracy)

    def _optimizer(self):
        self.global_step = tf.get_variable('global_step',dtype=tf.int32,initializer=tf.constant(0),trainable=False)
        learning_rate = tf.train.exponential_decay( self.para.learningRate,#初始学习率
                                                    self.global_step,#Variable，每batch加一
                                                    self.para.lr_decay_steps,#global_step/decay_steps得到decay_rate的幂指数
                                                    self.para.lr_decay_rate,#学习率衰减系数
                                                    staircase=False)#若True ，则学习率衰减呈离散间隔

        self.learning_rate = tf.maximum(learning_rate, self.para.minimum_lr)
        tf.summary.scalar("learning_rate", self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt_op = self.optimizer.minimize(self.loss,global_step=self.global_step)

    def _build(self):
        self.architecture_1()

    def architecture_1(self):
        # shape of GraphConv input and output data as batch_size * point_num * feature_dim
        # -----------------------------------------------STN-----------------------------------------------------------
        if (self.para.useSTN):
            self.layers.append(STN(is_training=self.is_training, logging=self.logging))
        # ----------------------------------------------gcn layer 1----------------------------------------------------

        self.layers.append(ParallelBlock(graph=self.other_inputs['graph_1'],
                                         input_dim=[1, 2],
                                         output_dim=[32, 32],
                                         chebyshevOrder=3,
                                         drop_out=self.para.keep_prob_1,
                                         batch_index=self.other_inputs['poolIndex_1'],
                                         batch_size=self.batch_size,
                                         clusterNumber=self.para.vertexNumG2,
                                         nearestNeighbor=self.para.poolNumG1,
                                         is_training=self.is_training,
                                         logging=self.logging,
                                         ))
        # -----------------------------------------------STN-----------------------------------------------------------
        # if (self.para.useSTN):
        #     self.layers.append(STN(is_training=self.is_training, logging=self.logging))
        # -------------------------------------------ChannelAttention 1------------------------------------------------
        if (self.para.useChannelAttention):
            self.layers.append(ChannelAttention(input_dim=self.para.gcn_1_filter_n, is_training=self.is_training,logging=self.logging))
        # ----------------------------------------------gcn layer 2----------------------------------------------------
        self.layers.append(ParallelBlock(graph=self.other_inputs['graph_2'],
                                         input_dim=[32, 32],
                                         output_dim=[64, 64],
                                         chebyshevOrder=3,
                                         drop_out=self.para.keep_prob_1,
                                         batch_index=self.other_inputs['poolIndex_2'],
                                         batch_size=self.batch_size,
                                         clusterNumber=self.para.vertexNumG3,
                                         nearestNeighbor=self.para.poolNumG2,
                                         is_training=self.is_training,
                                         logging=self.logging,
                                         ))
        # -------------------------------------------ChannelAttention 2------------------------------------------------
        if (self.para.useChannelAttention):
            self.layers.append(ChannelAttention(input_dim=self.para.gcn_2_filter_n, is_training=self.is_training,logging=self.logging))
        # ----------------------------------------------gcn layer 3----------------------------------------------------
        self.layers.append(ParallelBlock(graph=self.other_inputs['graph_3'],
                                         input_dim=[64, 64],
                                         output_dim=[128, 128],
                                         chebyshevOrder=3,
                                         drop_out=self.para.keep_prob_1,
                                         batch_index=self.other_inputs['poolIndex_3'],
                                         batch_size=self.batch_size,
                                         clusterNumber=self.para.vertexNumG4,
                                         nearestNeighbor=self.para.poolNumG3,
                                         is_training=self.is_training,
                                         logging=self.logging,
                                         ))
        # -------------------------------------------ChannelAttention 3------------------------------------------------
        if (self.para.useChannelAttention):
            self.layers.append(ChannelAttention(input_dim=self.para.gcn_3_filter_n, is_training=self.is_training,logging=self.logging))
        # ----------------------------------------------FC layer 1-----------------------------------------------------
        self.layers.append(Dense(input_dim=2*self.para.gcn_3_filter_n * self.para.vertexNumG4,
                                 output_dim=self.para.fc_1_n,
                                 dropout=self.para.keep_prob_2,
                                 input_reshape=True,
                                 act=tf.nn.relu,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))  # fc layer 1
        # ----------------------------------------------FC layer 2-----------------------------------------------------
        self.layers.append(Dense(input_dim=self.para.fc_1_n,
                                 output_dim=self.para.fc_2_n,
                                 dropout=self.para.keep_prob_2,
                                 input_reshape=False,
                                 act=tf.nn.relu,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))  # fc layer 2
        # ----------------------------------------------FC layer 3-----------------------------------------------------
        self.layers.append(Dense(input_dim=self.para.fc_2_n,
                                 output_dim=self.para.outputClassN,
                                 dropout=self.para.keep_prob_2,
                                 input_reshape=False,
                                 act=lambda x: x,
                                 bias=True,
                                 bn=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))  # fc layer 2

    def architecture_2(self):
        # [B,512,3]
        # -----------------------------------------------STN-----------------------------------------------------------
        self.layers.append(TransformNet(input_dim=1,K=3,reshape=True,is_training=self.is_training,logging=self.logging))
        # ----------------------------------------------gcn layer 1----------------------------------------------------
        # [B,512,3]
        self.layers.append(GraphConv(graph=self.other_inputs['graph_1'],
                                     input_dim=self.para.input_data_dim,
                                     output_dim=self.para.gcn_1_filter_n,
                                     chebyshevOrder=self.para.chebyshev_1_Order,
                                     dropout=self.para.keep_prob_1,
                                     bn=True,
                                     bias=True,
                                     act=tf.nn.relu,
                                     is_training=self.is_training,
                                     logging=self.logging
                                     ))
        # [B,512,32]
        self.layers.append(GraphPool(batch_index=self.other_inputs['poolIndex_1'],
                                    batch_size=self.batch_size,
                                    clusterNumber=self.para.vertexNumG2,
                                    nearestNeighbor=self.para.poolRangeG1,
                                    logging=self.logging,
                                    pool=tf.reduce_max
                                    ))
        # [B,256,32]
        # -----------------------------------------------STN-----------------------------------------------------------
        self.layers.append(TransformNet(input_dim=1,K=32,reshape=True,is_training=self.is_training,logging=self.logging))
        # ----------------------------------------------gcn layer 2----------------------------------------------------
        # [B,256,32]
        self.layers.append(GraphConv(graph=self.other_inputs['graph_2'],
                                     input_dim=self.para.gcn_1_filter_n,
                                     output_dim=self.para.gcn_2_filter_n,
                                     chebyshevOrder=self.para.chebyshev_2_Order,
                                     dropout=self.para.keep_prob_1,
                                     bn=True,
                                     bias=True,
                                     act=tf.nn.relu,
                                     is_training=self.is_training,
                                     logging=self.logging
                                     ))
        # [B,256,64]
        self.layers.append(GraphPool(batch_index=self.other_inputs['poolIndex_2'],
                                    batch_size=self.batch_size,
                                    clusterNumber=self.para.vertexNumG3,
                                    nearestNeighbor=self.para.poolRangeG2,
                                    logging=self.logging,
                                    pool=tf.reduce_max
                                    ))
        # [B,64,64]
        # ----------------------------------------------gcn layer 3----------------------------------------------------
        self.layers.append(GraphConv(graph=self.other_inputs['graph_3'],
                                     input_dim=self.para.gcn_2_filter_n,
                                     output_dim=self.para.gcn_3_filter_n,
                                     chebyshevOrder=self.para.chebyshev_3_Order,
                                     dropout=self.para.keep_prob_1,
                                     bn=True,
                                     bias=True,
                                     act=tf.nn.relu,
                                     is_training=self.is_training,
                                     logging=self.logging
                                     ))
        # [B,64,128]
        self.layers.append(GraphPool(batch_index=self.other_inputs['poolIndex_3'],
                                    batch_size=self.batch_size,
                                    clusterNumber=self.para.vertexNumG4,
                                    nearestNeighbor=self.para.poolRangeG3,
                                    logging=self.logging,
                                    pool=tf.reduce_max
                                    ))
        # [B,32,128]
        # ----------------------------------------------FC layer 1-----------------------------------------------------
        self.layers.append(Dense(input_dim=self.para.gcn_3_filter_n * self.para.vertexNumG4,
                                 output_dim=self.para.fc_1_n,
                                 dropout=self.para.keep_prob_2,
                                 input_reshape=True,
                                 act=tf.nn.relu,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))  # fc layer 1
        # ----------------------------------------------FC layer 2-----------------------------------------------------
        self.layers.append(Dense(input_dim=self.para.fc_1_n,
                                 output_dim=self.para.fc_2_n,
                                 dropout=self.para.keep_prob_2,
                                 input_reshape=False,
                                 act=tf.nn.relu,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))  # fc layer 2
        # ----------------------------------------------FC layer 3----------------------------------------------------
        self.layers.append(Dense(input_dim=self.para.fc_2_n,
                                 output_dim=self.para.fc_3_n,
                                 dropout=self.para.keep_prob_2,
                                 input_reshape=False,
                                 act=tf.nn.relu,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))  # fc layer 2
        # ----------------------------------------------FC layer 4----------------------------------------------------
        self.layers.append(Dense(input_dim=self.para.fc_3_n,
                                 output_dim=self.para.outputClassN,
                                 dropout=self.para.keep_prob_2,
                                 input_reshape=False,
                                 act=lambda x: x,
                                 bias=True,
                                 bn=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))  # fc layer 2