from layers import *

class Block(object):
    # class variable
    def __init__(self,**kwargs):
        allowed_kwargs = {'is_training','name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            block = self.__class__.__name__.lower()
            name = block + '_' + str(get_layer_uid(block))
        self.name = name
        self.logging = kwargs.get('logging', False)
        self.is_training = kwargs.get('is_training', True)
        self.block = []
        self.activations = []
        Block.global_variable_scope = tf.get_variable_scope()

    def __call__(self, inputs): #外层运行主体
        with tf.name_scope(self.name):
            outputs = self._call(inputs)#内层运行主体
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def build(self):
        with tf.variable_scope(self.name + '_var'):
            self._build()

    def _build(self):
        raise NotImplementedError

    def _call(self,inputs):
        return inputs

class STN(Block):
    def __init__(self,**kwargs):
        super(STN, self).__init__(**kwargs)
        self.build()

    def _build(self):
        self.block.append(GlobalPooling(transpose=True,use_avg=True,use_max=True,use_min=True))
        # (B,1,C,3)---[theta,phi][avg,max,min]
        self.block.append(Conv2d(input_dim=3,
                                 output_dim=3 * 16,
                                 kernel_size=[1, 1],
                                 dropout=1.0,
                                 bn=False,
                                 bias=False,
                                 input_reshape=False,
                                 act=lambda x: x,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))

        self.block.append(Conv2d(input_dim=3 * 16,
                                 output_dim=1,
                                 kernel_size=[1, 1],
                                 dropout=1.0,
                                 bn=False,
                                 bias=False,
                                 input_reshape=False,
                                 act=tf.nn.tanh,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        # (B,1,C,1)---[theta,phi][recoup]
    def _call(self,inputs):
        inputs_r = inputs[:, :, 0:1]
        inputs_angle = inputs[:, :, 1:]     #(B,N,2)
        self.activations.append(inputs_angle)
        for block in self.block:
            hidden = block(self.activations[-1])
            self.activations.append(hidden)
        self.transform = self.activations[-1]
        with tf.name_scope('recovery_op'):
            self.transform = tf.reduce_sum(self.transform, axis=-1)     # (B,1,C)
            # self.transform = tf.nn.tanh(self.transform)
            outputs_angle = tf.add(self.transform, inputs_angle)
            outputs = tf.concat([inputs_r, outputs_angle], axis=2)
        return outputs

class ChannelAttention(Block):
    def __init__(self, input_dim, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.build()

    def _build(self):
        self.block.append(GlobalPooling(transpose=False,use_avg=True,use_max=True))
        # (B,1,2,C)
        self.block.append(Conv2d(input_dim=self.input_dim,
                                 output_dim=self.input_dim//8,
                                 kernel_size=[1,1],
                                 dropout=1.0,
                                 bn=False,
                                 bias=False,
                                 input_reshape=False,
                                 act= lambda x:x,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))

        self.block.append(Conv2d(input_dim=self.input_dim//8,
                                 output_dim=self.input_dim,
                                 kernel_size=[1,1],
                                 dropout=1.0,
                                 bn=False,
                                 bias=False,
                                 input_reshape=False,
                                 act= lambda x:x,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))

    def _call(self,inputs):
        self.activations.append(inputs)
        for block in self.block:
            hidden = block(self.activations[-1])
            self.activations.append(hidden)
        self.scale = self.activations[-1]
        with tf.name_scope('channel_wise_multiply'):
            self.scale = tf.reduce_sum(self.scale,axis=2)
            self.scale = tf.nn.sigmoid(self.scale)
            outputs = tf.multiply(self.scale,inputs)
        return outputs

