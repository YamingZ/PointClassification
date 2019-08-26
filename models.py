from layers import *
from block import *

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name+'_var',reuse=tf.AUTO_REUSE):
            self._build()

            # Build sequential layer model
            self.activations.append(self.inputs)
            for layer in self.layers:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
            self.outputs = self.activations[-1]

            # Build metrics
            with tf.name_scope("Loss"):
                self._loss()
            with tf.name_scope("Accuracy"):
                self._accuracy()
            with tf.name_scope("Optimizer"):
                self._optimizer()

            # Store model variables for easy access
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.vars = {var.name: var for var in variables}

            self.summary = tf.summary.merge_all()

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def _optimizer(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GPN(Model):
    def __init__(self,para,placeholders,**kwargs):
        super(GPN,self).__init__(**kwargs)
        self.para = para
        self.is_training = placeholders['isTraining']
        self.batch_size = placeholders['batch_size']
        # signal training device
        if self.para.GpuNums == 1:
            self.inputs = placeholders['coordinate']
            self.other_inputs = placeholders
            self.build()

    def dataInput(self,coordinate,label,weights,graph_1,graph_2,graph_3,batch_index_l1,batch_index_l2):
        self.inputs = coordinate
        self.other_inputs = {'label': label,
                             'weights': weights,
                             'graph_1': graph_1,
                             'graph_2': graph_2,
                             'graph_3': graph_3,
                             'batch_index_l1': batch_index_l1,
                             'batch_index_l2': batch_index_l2}
        self.build()

    def tower_loss(self,scope):
        losses = tf.get_collection('losses', scope)
        total_loss = tf.add_n(losses, name='total_loss')
        return total_loss

    def _loss(self):
        with tf.name_scope('L2_loss'):
            vars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'weights' in v.name]) * self.para.l2_rate
        tf.summary.scalar("l2_loss", l2_loss)
        # with tf.name_scope('TMat_loss'):
        #     theta = self.layers[0].outputs_angle_theta
        #     phi = self.layers[0].outputs_angle_phi
        #     diff = tf.nn.relu(theta**2-theta*np.pi) + tf.nn.relu(phi**2-phi*np.pi*2)
        #     mat_diff_loss = tf.nn.l2_loss(diff) * self.para.tmat_rate

        #     self.transform_matrix = self.layers[0].transform
        #     mat_diff = tf.matmul(self.transform_matrix, tf.transpose(self.transform_matrix, perm=[0, 2, 1]))
        #     mat_diff -= tf.constant(np.eye(3), dtype=tf.float32)
        #     mat_diff_loss = tf.nn.l2_loss(mat_diff) * self.para.tmat_rate
        # tf.summary.scalar('TMat_loss', mat_diff_loss)
        with tf.name_scope('Loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.other_inputs['label'])
            loss = tf.multiply(loss, self.other_inputs['weights'])
            loss = tf.reduce_mean(loss)
            self.loss = loss + l2_loss #+ mat_diff_loss
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
        learning_rate = tf.train.exponential_decay(self.para.learningRate,#初始学习率
                                                        self.global_step,#Variable，每batch加一
                                                        self.para.lr_decay_steps,#global_step/decay_steps得到decay_rate的幂指数
                                                        0.96,#学习率衰减系数
                                                        staircase=False)#若True ，则学习率衰减呈离散间隔
        self.learning_rate = tf.maximum(learning_rate, self.para.minimum_lr)
        tf.summary.scalar("learning_rate", self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt_op = self.optimizer.minimize(self.loss,global_step=self.global_step)


    def _build(self):
        # batch_size * point_num * feature_dim * 1
        if(self.para.useSTN):
            self.layers.append(STN(transform_dim=2,is_training=self.is_training,logging=self.logging))  #STN layer
        self.layers.append(GraphConv(graph=self.other_inputs['graph_1'],
                                     input_dim=self.para.input_data_dim,
                                     output_dim=self.para.gcn_1_filter_n,
                                     pointnumber=self.para.pointNumber,
                                     chebyshevOrder=self.para.chebyshev_1_Order,
                                     dropout=self.para.keep_prob_1,
                                     bn=True,
                                     act=tf.nn.relu,
                                     is_training=self.is_training,
                                     logging=self.logging
                                     )
                           ) # gcn layer 1
        # batch_size * point_num * 1 * feature_dim
        self.layers.append(GraphMaxPool(batch_index=self.other_inputs['batch_index_l1'],
                                        batch_size=self.batch_size,
                                        clusterNumber=self.para.clusterNumberL1,
                                        nearestNeighbor=self.para.nearestNeighborL1,
                                        featuredim=self.para.gcn_1_filter_n,
                                        logging=self.logging
                                        )
                           ) # max pooling
        self.layers.append(GraphConv(graph=self.other_inputs['graph_2'],
                                     input_dim=self.para.gcn_1_filter_n,
                                     output_dim=self.para.gcn_2_filter_n,
                                     pointnumber=self.para.clusterNumberL1,
                                     chebyshevOrder=self.para.chebyshev_2_Order,
                                     dropout=self.para.keep_prob_1,
                                     bn=True,
                                     act=tf.nn.relu,
                                     is_training=self.is_training,
                                     logging=self.logging
                                     )
                           ) # gcn layer 2
        self.layers.append(GraphMaxPool(batch_index=self.other_inputs['batch_index_l2'],
                                        batch_size=self.batch_size,
                                        clusterNumber=self.para.clusterNumberL2,
                                        nearestNeighbor=self.para.nearestNeighborL2,
                                        featuredim=self.para.gcn_2_filter_n,
                                        logging=self.logging
                                        )
                           ) # max pooling

        self.layers.append(GraphConv(graph=self.other_inputs['graph_3'],
                                     input_dim=self.para.gcn_2_filter_n,
                                     output_dim=self.para.gcn_3_filter_n,
                                     pointnumber=self.para.clusterNumberL2,
                                     chebyshevOrder=self.para.chebyshev_3_Order,
                                     dropout=self.para.keep_prob_1,
                                     bn=True,
                                     act=tf.nn.relu,
                                     pooling= True,
                                     is_training=self.is_training,
                                     logging=self.logging
                                     )
                           ) # gcn layer 2
        # 28 * 1 * 1 * 1024 --> 28 * 1024
        self.layers.append(Dense(input_dim=self.para.gcn_3_filter_n,
                                 output_dim=self.para.fc_1_n,
                                 dropout=self.para.keep_prob_2,
                                 input_reshape=False,
                                 act= tf.nn.relu,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 )
                           )  # fc layer 1
        # 28 * 128
        self.layers.append(Dense(input_dim=self.para.fc_1_n,
                                 output_dim=self.para.outputClassN,
                                 dropout=self.para.keep_prob_2,
                                 input_reshape=False,
                                 act=lambda x: x,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 )
                           )     # fc layer 3
        # 28 * 40

