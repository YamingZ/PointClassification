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
    def __init__(self,input_dim=None,**kwargs):
        super(STN, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.build()
    def _build(self):
        self.block.append(GlobalPooling(transpose=True,use_avg=True,use_max=True,use_min=True))
        # (B,1,C,3)---[theta,phi][avg,max,min]
        self.block.append(Conv2d(input_dim=3,
                                 output_dim=16,
                                 kernel_size=[1, 1],
                                 dropout=1.0,
                                 bn=False,
                                 bias=True,
                                 input_reshape=False,
                                 act=lambda x: x,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.block.append(Conv2d(input_dim=16,
                                 output_dim=16,
                                 kernel_size=[1, 1],
                                 dropout=1.0,
                                 bn=False,
                                 bias=True,
                                 input_reshape=False,
                                 act=lambda x: x,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.block.append(Conv2d(input_dim=16,
                                 output_dim=1,
                                 kernel_size=[1, 1],
                                 dropout=1.0,
                                 bn=False,
                                 bias=True,
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
            outputs_angle = tf.add(self.transform, inputs_angle)
            outputs = tf.concat([inputs_r, outputs_angle], axis=2)      # (B,n,C)
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


class ParallelBlock(Block):
    def __init__(self,graph,input_dim,output_dim,chebyshevOrder,drop_out,batch_index,batch_size,clusterNumber,nearestNeighbor, **kwargs):
        super(ParallelBlock,self).__init__(**kwargs)
        # GraphConv
        self.graph = graph
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.chebyshevOrder = chebyshevOrder
        self.drop_out = drop_out
        # GrapPool
        self.batch_index = batch_index
        self.batch_size = batch_size
        self.clusterNumber = clusterNumber
        self.nearestNeighbor = nearestNeighbor
        self.build()

    def _build(self):
        # -----------------------------------for R---------------------------------------
        self.block.append(GraphConv(graph=self.graph,
                                     input_dim=self.input_dim[0],
                                     output_dim=self.output_dim[0],
                                     chebyshevOrder=self.chebyshevOrder,
                                     dropout=self.drop_out,
                                     bn=False,
                                     bias=True,
                                     act=tf.nn.relu,
                                     is_training=self.is_training,
                                     logging=self.logging
                                     )
                           )
        self.block.append(GraphPool(batch_index=self.batch_index,
                                    batch_size=self.batch_size,
                                    clusterNumber=self.clusterNumber,
                                    nearestNeighbor=self.nearestNeighbor,
                                    logging=self.logging,
                                    pool=tf.reduce_max
                                    )
                           )
        #-----------------------------------for theta phi---------------------------------------
        self.block.append(GraphConv(graph=self.graph,
                                     input_dim=self.input_dim[1],
                                     output_dim=self.output_dim[1],
                                     chebyshevOrder=self.chebyshevOrder,
                                     dropout=self.drop_out,
                                     bn=False,
                                     bias=True,
                                     act=tf.nn.sigmoid,
                                     is_training=self.is_training,
                                     logging=self.logging
                                     )
                           )
        self.block.append(GraphPool(batch_index=self.batch_index,
                                    batch_size=self.batch_size,
                                    clusterNumber=self.clusterNumber,
                                    nearestNeighbor=self.nearestNeighbor,
                                    logging=self.logging,
                                    pool=tf.reduce_mean
                                    )
                           )

    def _call(self,inputs):
        inputs_r = inputs[:, :, 0 : self.input_dim[0]]
        inputs_angle = inputs[:, :, self.input_dim[0] :]     #(B,N,2)

        outputs_r = self.block[0](inputs_r)
        outputs_r = self.block[1](outputs_r)
        outputs_angle = self.block[2](inputs_angle)
        outputs_angle = self.block[3](outputs_angle)
        outputs = tf.concat([outputs_r, outputs_angle], axis=2)
        return outputs



class TransformNet(Block):
    def __init__(self,input_dim,K,reshape=False,**kwargs):
        super(TransformNet, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.K = K
        self.reshape = reshape
        self.build()

    def _build(self):
        self.block.append(Conv2d(input_dim=self.input_dim,
                                 output_dim=64,
                                 kernel_size=[1, self.K] if self.reshape else [1,1],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 input_reshape=False,
                                 act=tf.nn.relu,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.block.append(Conv2d(input_dim=64,
                                 output_dim=128,
                                 kernel_size=[1, 1],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 input_reshape=False,
                                 act=tf.nn.relu,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.block.append(Conv2d(input_dim=128,
                                 output_dim=1024,
                                 kernel_size=[1, 1],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 input_reshape=False,
                                 act=tf.nn.relu,
                                 pooling=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.block.append(Dense(input_dim=1024,
                                 output_dim=512,
                                 dropout=1.0,
                                 input_reshape=True,
                                 act=tf.nn.relu,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.block.append(Dense(input_dim=512,
                                 output_dim=256,
                                 dropout=1.0,
                                 input_reshape=False,
                                 act=tf.nn.relu,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.block.append(MatmulLayer(input_dim=256,
                                      K=self.K,
                                      logging=self.logging
                                      ))


    def _call(self,inputs):
        if self.reshape:
            inputs = tf.expand_dims(inputs, -1)
        self.activations.append(inputs)
        for block in self.block:
            hidden = block(self.activations[-1])
            self.activations.append(hidden)
        self.transform = self.activations[-1]
        if self.reshape:
            net_transformed = tf.matmul(tf.squeeze(inputs, axis=[3]), self.transform)
            # net_transformed = tf.expand_dims(net_transformed, -1)
        else:
            net_transformed = tf.matmul(tf.squeeze(inputs, axis=[2]), self.transform)
            # net_transformed = tf.expand_dims(net_transformed, [2])

        return net_transformed

class STN_ADD(Block):
    def __init__(self, feature_dim, fix_feature_dim,  **kwargs):
        super(STN_ADD, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.fix_feature_dim = fix_feature_dim
        self.build()

    def _build(self):
        self.block.append(Conv2d(input_dim=1,
                                 output_dim=64,
                                 kernel_size=[1,self.fix_feature_dim],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 input_reshape=True,
                                 act=tf.nn.relu,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.block.append(Conv2d(input_dim=64,
                                 output_dim=128,
                                 kernel_size=[1, 1],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 input_reshape=False,
                                 act=tf.nn.relu,
                                 pooling=False,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.block.append(Conv2d(input_dim=128,
                                 output_dim=1024,
                                 kernel_size=[1, 1],
                                 dropout=1.0,
                                 bn=True,
                                 bias=True,
                                 input_reshape=False,
                                 act=tf.nn.relu,
                                 pooling=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.block.append(Dense(input_dim=1024,
                                 output_dim=512,
                                 dropout=1.0,
                                 input_reshape=True,
                                 act=tf.nn.relu,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.block.append(Dense(input_dim=512,
                                 output_dim=256,
                                 dropout=1.0,
                                 input_reshape=False,
                                 act=tf.nn.relu,
                                 bias=True,
                                 bn=True,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))
        self.block.append(Dense(input_dim=256,
                                 output_dim=self.fix_feature_dim,
                                 dropout=1.0,
                                 input_reshape=False,
                                 bn=True,
                                 bias=True,
                                 act=tf.nn.tanh,
                                 is_training=self.is_training,
                                 logging=self.logging
                                 ))

    def _call(self,inputs):
        inputs_r = inputs[:, :, 0:self.feature_dim-self.fix_feature_dim]
        inputs_angle = inputs[:, :, self.feature_dim-self.fix_feature_dim:]     #(B,N,64)

        self.activations.append(inputs_angle)
        for block in self.block:
            hidden = block(self.activations[-1])
            self.activations.append(hidden)
        self.correction = self.activations[-1]

        outputs_angle = tf.add(tf.expand_dims(self.correction, [1]), inputs_angle)
        outputs = tf.concat([inputs_r, outputs_angle], axis=2)  # (B,n,C)

        return outputs
