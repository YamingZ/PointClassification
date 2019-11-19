from models import *

class PointNet(Model):
    def __init__(self, para, placeholders, **kwargs):
        super(PointNet, self).__init__(**kwargs)
        self.para = para
        self.is_training = placeholders['isTraining']
        self.batch_size = placeholders['batch_size']
        # signal training device
        self.inputs = placeholders['coordinate']
        self.other_inputs = placeholders
        self.build()

    def _loss(self):
        with tf.name_scope('MatLoss'):
            self.transform = self.layers[0].transform
            K = self.transform.get_shape()[1].value
            mat_diff = tf.matmul(self.transform, tf.transpose(self.transform, perm=[0, 2, 1]))
            mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
            mat_diff_loss = tf.nn.l2_loss(mat_diff)
        tf.summary.scalar('MatLoss', mat_diff_loss)
        with tf.name_scope('Loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs, labels=self.other_inputs['label'])
            loss = tf.reduce_mean(loss)
        tf.summary.scalar("softmax_loss", loss)
        self.loss = loss + mat_diff_loss

    def _accuracy(self):
        self.probability = tf.nn.softmax(self.outputs)
        self.predictLabels = tf.argmax(self.probability, axis=1)  # softmax can be delete
        correct_prediction = tf.equal(self.predictLabels, tf.argmax(self.other_inputs['label'], axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("acc", self.accuracy)

    def _optimizer(self):
        self.global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=tf.constant(0), trainable=False)
        learning_rate = tf.train.exponential_decay(self.para.learningRate,  # 初始学习率
                                                   self.global_step,  # Variable，每batch加一
                                                   self.para.lr_decay_steps,  # global_step/decay_steps得到decay_rate的幂指数
                                                   self.para.lr_decay_rate,  # 学习率衰减系数
                                                   staircase=False)  # 若True ，则学习率衰减呈离散间隔

        self.learning_rate = tf.maximum(learning_rate, self.para.minimum_lr)
        tf.summary.scalar("learning_rate", self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def _build(self):
        self.layers.append(TransformNet(input_dim=1,K=3,reshape=True,is_training=self.is_training,logging=self.logging))
        self.layers.append(Conv2d(input_dim=1,
                                 output_dim=64,
                                 kernel_size=[1,3],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 input_reshape=False,
                                 act= tf.nn.relu,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.layers.append(Conv2d(input_dim=64,
                                 output_dim=64,
                                 kernel_size=[1,1],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 input_reshape=False,
                                 act= tf.nn.relu,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.layers.append(TransformNet(input_dim=64,K=64,reshape=False,is_training=self.is_training,logging=self.logging))
        self.layers.append(Conv2d(input_dim=64,
                                 output_dim=64,
                                 kernel_size=[1,1],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 input_reshape=False,
                                 act= tf.nn.relu,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.layers.append(Conv2d(input_dim=64,
                                 output_dim=128,
                                 kernel_size=[1,1],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 input_reshape=False,
                                 act= tf.nn.relu,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.layers.append(Conv2d(input_dim=128,
                                 output_dim=1024,
                                 kernel_size=[1,1],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 input_reshape=False,
                                 act= tf.nn.relu,
                                 pooling=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.layers.append(Dense(input_dim=1024,
                                 output_dim=512,
                                 dropout=0.7,
                                 input_reshape=True,
                                 act=tf.nn.relu,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.layers.append(Dense(input_dim=512,
                                 output_dim=256,
                                 dropout=0.7,
                                 input_reshape=True,
                                 act=tf.nn.relu,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.layers.append(Dense(input_dim=256,
                                 output_dim=40,
                                 dropout=1.0,
                                 input_reshape=True,
                                 act=lambda x: x,
                                 bias=True,
                                 bn=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))