import tensorflow as tf
from utils import *
# from sampling.tf_sampling import *
# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


# 实现Batch Normalization
def batch_norm(inputs,offset,scale,is_training,moving_decay=0.9,eps=1e-5):
    axes = list(range(len(inputs.get_shape()) - 1))
    # 计算当前整个batch的均值与方差
    batch_mean, batch_var = tf.nn.moments(inputs,axes,name='moments')
    # 采用滑动平均更新均值与方差
    ema = tf.train.ExponentialMovingAverage(moving_decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean,batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
    # with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    mean, var = tf.cond(is_training,mean_var_with_update,lambda:(ema.average(batch_mean),ema.average(batch_var)))
    # 最后执行batch normalization
    return tf.nn.batch_normalization(inputs,mean,var,offset,scale,eps)

#dropout add istraining param
def dropout(input,keep_prob,is_training):
    return tf.cond(is_training,lambda :tf.nn.dropout(input, keep_prob=keep_prob),lambda :input)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs): #外层运行主体
        with tf.name_scope(self.name):
            outputs = self._call(inputs)#内层运行主体
            if self.logging:
                tf.summary.histogram('inputs', inputs)
                tf.summary.histogram('outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphConv(Layer):
    def __init__(self,graph,input_dim,output_dim,pointnumber,chebyshevOrder,dropout, bn=False, bias=True, act=tf.nn.relu,pooling=False,is_training=True,**kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.graph = graph
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pointnumber = pointnumber
        self.chebyOrder = chebyshevOrder

        self.bn = bn
        self.dropout = dropout
        self.pooling = pooling
        self.bias = bias
        self.act = act

        self.is_training = is_training

        #initialized variable
        self.initVariable()

        if self.logging:
            self._log_vars()

    def initVariable(self):
        with tf.variable_scope(self.name + '_vars'):
            for i in range(self.chebyOrder):
                self.vars['weights_'+ str(i)] = tf.get_variable('weights_'+str(i),[self.input_dim,self.output_dim],initializer=tf.glorot_uniform_initializer())
            if self.bias:
                self.vars['bias'] = tf.get_variable('bias',[self.output_dim],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.05))
            if self.bn:
                with tf.variable_scope(self.name + '_bn_vars'):
                    self.vars['offset'] = tf.get_variable('offset', [self.output_dim],initializer=tf.constant_initializer(0))
                    self.vars['scale'] = tf.get_variable('scale', [self.output_dim],initializer=tf.constant_initializer(1))

    def _call(self, coordinate):
        with tf.name_scope('cheby_op'):
            scaledLaplacian = tf.reshape(self.graph, [-1, self.pointnumber, self.pointnumber])
            chebyPoly = []  # Chebyshev polynomials
            cheby_K_Minus_1 = tf.matmul(scaledLaplacian, coordinate)
            cheby_K_Minus_2 = coordinate

            chebyPoly.append(cheby_K_Minus_2)
            chebyPoly.append(cheby_K_Minus_1)
            for i in range(2, self.chebyOrder):
                chebyK = 2 * tf.matmul(scaledLaplacian, cheby_K_Minus_1) - cheby_K_Minus_2
                chebyPoly.append(chebyK)
                cheby_K_Minus_2 = cheby_K_Minus_1
                cheby_K_Minus_1 = chebyK

            chebyOutput = []
            for i in range(self.chebyOrder):
                weights = self.vars['weights_' + str(i)]
                chebyPolyReshape = tf.reshape(chebyPoly[i], [-1, self.input_dim])
                output = tf.matmul(chebyPolyReshape, weights)
                output = tf.reshape(output, [-1, self.pointnumber, self.output_dim])
                chebyOutput.append(output)
        # cheby_op不改变节点数，只改变每个节点的特征数，3--->1000
        if self.bias:
            with tf.name_scope("add_bias"):
                gcn_output = tf.add_n(chebyOutput) + self.vars['bias']
        else:
            gcn_output = chebyOutput
        # batch normalization
        if self.bn:
            with tf.name_scope('batch_norm'):
                gcn_output = batch_norm(gcn_output,self.vars['offset'],self.vars['scale'],is_training=self.is_training)
        with tf.name_scope("activate"):
            gcn_output = self.act(gcn_output)

        with tf.name_scope("dropout"):
                gcn_output = dropout(gcn_output,self.dropout,self.is_training)
        if self.pooling:
            with tf.name_scope("max_pooling"):
                # 每个输出维度中取所有点中数值最大的点输出
                gcn_output = tf.reduce_mean(gcn_output, axis=1)
        return gcn_output

class GraphMaxPool(Layer):
    def __init__(self,batch_size,batch_index,clusterNumber,nearestNeighbor,featuredim,**kwargs):
        super(GraphMaxPool,self).__init__(**kwargs)
        self.batch_index = batch_index
        self.batch_size = batch_size
        self.clusterNumber = clusterNumber
        self.nearestNeighbor = nearestNeighbor
        self.featuredim = featuredim

    def _call(self, inputs):
        M = self.clusterNumber
        k = self.nearestNeighbor
        n = self.featuredim
        batch_size = self.batch_size
        batch_index = self.batch_index

        index_reshape = tf.reshape(batch_index, [M * k * batch_size, 1])
        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1))
        batch_idx_tile = tf.tile(batch_idx, (1, M * k))
        batch_idx_tile_reshape = tf.reshape(batch_idx_tile, [M * k * batch_size, 1])
        new_index = tf.concat([batch_idx_tile_reshape, index_reshape], axis=1)
        group_features = tf.gather_nd(inputs, new_index)  # get M*K points' feature form N points' feature
        group_features_reshape = tf.reshape(group_features, [batch_size, M, k, n])
        '''That operation is often taken to be the maximum value, but it can be any permutation invariant operation, such as a sum or an average.'''
        max_features = tf.reduce_max(group_features_reshape, axis=2)
        return max_features


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout, input_reshape=False,act=tf.nn.relu, bias=False,bn=False,is_training=True,**kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.bn = bn
        self.bias = bias
        self.is_training = is_training

        self.input_reshape = input_reshape
        self.input_dim = input_dim
        self.output_dim = output_dim

        #initialized variable
        self.initVariable()

        if self.logging:
            self._log_vars()

    def initVariable(self):
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable('weights',[self.input_dim,self.output_dim],
                                                   initializer=tf.glorot_uniform_initializer())
            if self.bias:
                self.vars['bias'] = tf.get_variable('bias',[self.output_dim],
                                                    initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.05))
            if self.bn:
                with tf.variable_scope(self.name + '_bn_vars'):
                    self.vars['offset'] = tf.get_variable('offset', [self.output_dim],initializer=tf.constant_initializer(0))
                    self.vars['scale'] = tf.get_variable('scale', [self.output_dim],initializer=tf.constant_initializer(1))

    def _call(self, inputs):
        if self.input_reshape:
            x = tf.reshape(inputs,[-1,self.input_dim])
        else: x = inputs
        # dropout
        with tf.name_scope('dropout'):
            x = dropout(x,self.dropout,self.is_training)
        # transform
        with tf.name_scope('dot'):
            output = tf.matmul(x, self.vars['weights'])
        # bias
        if self.bias:
            with tf.name_scope('add_bias'):
                output += self.vars['bias']
        # batch normalization
        if self.bn:
            with tf.name_scope('batch_norm'):
                output = batch_norm(output,self.vars['offset'],self.vars['scale'],is_training=self.is_training)

        # activate
        with tf.name_scope('activate'):
            output = self.act(output)
        return output


class Conv2d(Layer):
    def __init__(self, input_dim, output_dim,kernel_size, dropout, input_reshape=False,act=tf.nn.relu, bias=False,bn=False,pooling=False,is_training=True,**kwargs):
        super(Conv2d, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.bn = bn
        self.bias = bias
        self.pooling = pooling
        self.is_training = is_training

        self.input_reshape = input_reshape
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        #initialized variable
        self.initVariable()
        if self.logging:
            self._log_vars()

    def initVariable(self):
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable('weights',[self.kernel_size[0],self.kernel_size[1],self.input_dim,self.output_dim],
                                                initializer=tf.glorot_uniform_initializer())
            if self.bias:
                self.vars['bias'] = tf.get_variable('bias',[self.output_dim],
                                                initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.05))
            if self.bn:
                with tf.variable_scope(self.name + '_bn_vars'):
                    self.vars['offset'] = tf.get_variable('offset', [self.output_dim],initializer=tf.constant_initializer(0))
                    self.vars['scale'] = tf.get_variable('scale', [self.output_dim],initializer=tf.constant_initializer(1))

    def _call(self, inputs):
        if self.input_reshape:
            x = tf.expand_dims(inputs, -1)
        else: x = inputs
        # dropout
        with tf.name_scope('dropout'):
            x = dropout(x, self.dropout,self.is_training)
        # convolution
        with tf.name_scope('convolution'):
            output = tf.nn.conv2d(x,self.vars['weights'],strides=[1,1,1,1],padding="VALID")
        # bias
        if self.bias:
            with tf.name_scope('add_bias'):
                output += self.vars['bias']
        # batch normalization
        if self.bn:
            with tf.name_scope('batch_norm'):
                output = batch_norm(output,self.vars['offset'],self.vars['scale'],is_training=self.is_training)

        # activate
        with tf.name_scope('activate'):
            output = self.act(output)

        if self.pooling:
            with tf.name_scope("max_pooling"):
                num_point = inputs.get_shape()[1].value
                output = tf.nn.avg_pool(output,
                                         ksize=[1, num_point, 1, 1],
                                         strides=[1, num_point, 1, 1],
                                         padding='VALID')
        return output


class Flatten(Layer):
    def __init__(self,**kwargs):
        super(Flatten, self).__init__(**kwargs)

    def _call(self,inputs):
        shape = inputs.get_shape()
        dim_num = 1
        for dim in shape[1:]:
            dim_num *= dim
        outputs = tf.reshape(inputs,[-1,dim_num])
        return outputs

class GlobalPooling(Layer):
    def __init__(self,**kwargs):
        super(GlobalPooling, self).__init__(**kwargs)

    def _call(self,inputs):
        num_point = inputs.get_shape()[1].value
        channel = inputs.get_shape()[2].value
        x = tf.expand_dims(inputs, -1)
        avgPool = tf.nn.avg_pool(x,ksize=[1, num_point, 1, 1],strides=[1, num_point, 1, 1],padding='VALID')
        avgPool = tf.reshape(avgPool,[-1, 1, 1, channel])
        maxPool = tf.nn.max_pool(x,ksize=[1, num_point, 1, 1],strides=[1, num_point, 1, 1],padding='VALID')
        maxPool = tf.reshape(maxPool, [-1, 1, 1, channel])
        output = tf.concat([avgPool,maxPool],axis=1)
        # print(output.get_shape())
        return output





