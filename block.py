from layers import *

class GlobalPooling(Layer):
    def __init__(self,**kwargs):
        super(GlobalPooling, self).__init__(**kwargs)

    def _call(self,inputs):
        num_point = inputs.get_shape()[1].value
        output = tf.nn.avg_pool(inputs,
                                ksize=[1, num_point, 1, 1],
                                strides=[1, num_point, 1, 1],
                                padding='VALID')
        return output


class TransformDense(Layer):
    def __init__(self,input_dim,transform_dim,**kwargs):
        super(TransformDense,self).__init__(**kwargs)
        self.input_dim = input_dim
        self.transform_dim = transform_dim

        # initialized variable
        with tf.variable_scope(self.name + '_vars'):
            output_dim = self.transform_dim * self.transform_dim
            self.vars['weights'] = tf.get_variable('weights', [self.input_dim,output_dim],
                                      initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.05),
                                      dtype=tf.float32)
            self.vars['biases'] = tf.get_variable('biases', [output_dim],
                                     initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.05),
                                     dtype=tf.float32)
            self.vars['biases'] += tf.constant(np.eye(self.transform_dim).flatten(), dtype=tf.float32)
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        transform = tf.matmul(inputs, self.vars['weights'])
        transform = tf.nn.bias_add(transform, self.vars['biases'])
        transform = tf.reshape(transform, [-1, self.transform_dim, self.transform_dim])
        return transform

class STN(object):
    # class variable
    global_variable_scope = None
    def __init__(self,transform_dim,is_training=True,**kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            block = self.__class__.__name__.lower()
            if STN.global_variable_scope == tf.get_variable_scope():
                name = block + '_' + str(get_layer_uid(block))
            else:
                name = block
        self.name = name
        self.logging = kwargs.get('logging', False)
        self.block = []
        self.activations = []

        self.transform_dim = transform_dim
        self.is_training = is_training

        with tf.variable_scope(self.name + '_var'):
            self._build()

        STN.global_variable_scope = tf.get_variable_scope()

    def _build(self):
        self.block.append(Conv2d(input_dim=1,
                                 output_dim=128,
                                 kernel_size=[1,self.transform_dim],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 input_reshape=True,
                                 act= tf.nn.relu,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        # self.block.append(Conv2d(input_dim=128,
        #                          output_dim=256,
        #                          kernel_size=[1,1],
        #                          dropout=1.0,
        #                          bn=True,
        #                          bias=True,
        #                          act=tf.nn.relu,
        #                          is_training=self.is_training,
        #                          logging=self.logging
        #                          ))
        self.block.append(Conv2d(input_dim=128,
                                 output_dim=512,
                                 kernel_size=[1,1],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 act=tf.nn.relu,
                                 pooling=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        # self.block.append(Dense(input_dim=512,
        #                         output_dim=256,
        #                         dropout=1.0,
        #                         input_reshape=True,
        #                         act= tf.nn.relu,
        #                         bias=True,
        #                         bn=True,
        #                         is_training=self.is_training,
        #                         logging=self.logging
        #                         ))
        self.block.append(Dense(input_dim=512,
                                output_dim=100,
                                dropout=1.0,
                                input_reshape=True,
                                act= tf.nn.relu,
                                bias=True,
                                bn=True,
                                is_training=self.is_training,
                                logging=self.logging
                                ))
        self.block.append(Dense(input_dim=100,
                                output_dim=self.transform_dim,
                                dropout=1.0,
                                input_reshape=False,
                                act= lambda x: x,
                                bias=True,
                                bn=True,
                                is_training=self.is_training,
                                logging=self.logging
                                ))

        # self.block.append(TransformDense(input_dim=256,
        #                                  transform_dim=self.transform_dim,
        #                                  logging=self.logging
        #                                  ))

    def _call(self, inputs):
        inputs_r = inputs[:,:,0:1]
        inputs_angle = inputs[:, :, 1:]
        inputs_angle_theta = inputs[:, :, 1:2]
        inputs_angle_phi = inputs[:, :, 2:3]
        self.activations.append(inputs_angle)
        for block in self.block:
            hidden = block(self.activations[-1])
            self.activations.append(hidden)
        self.transform = self.activations[-1]
        with tf.name_scope('recovery_op'):
            # outputs = tf.matmul(inputs, self.transform)
            self.outputs_angle_theta = inputs_angle_theta + tf.expand_dims(self.transform[:, 0:1], 1)
            self.outputs_angle_phi = inputs_angle_phi + tf.expand_dims(self.transform[:, 1:2], 1)
            outputs = tf.concat([inputs_r,self.outputs_angle_theta,self.outputs_angle_phi],axis=2)
        return outputs

    def __call__(self, inputs): #外层运行主体
        with tf.name_scope(self.name):
            outputs = self._call(inputs)#内层运行主体
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

class ChannelAttention(object):
    # class variable
    global_variable_scope = None
    def __init__(self, is_training=True, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            block = self.__class__.__name__.lower()
            if ChannelAttention.global_variable_scope == tf.get_variable_scope():
                name = block + '_' + str(get_layer_uid(block))
            else:
                name = block
        self.name = name
        self.logging = kwargs.get('logging', False)
        self.is_training = is_training
        self.block = []
        self.activations = []

        with tf.variable_scope(self.name + '_var'):
            self._build()

        ChannelAttention.global_variable_scope = tf.get_variable_scope()

    def __call__(self, inputs): #外层运行主体
        with tf.name_scope(self.name):
            outputs = self._call(inputs)#内层运行主体
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _build(self):
        pass

    def _call(self,inputs):
        pass

