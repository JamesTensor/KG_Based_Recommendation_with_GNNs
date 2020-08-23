# *_*coding:utf-8 *_*
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
#from tensorflow.python.keras.engine.training import Model
from tensorflow.keras import Model
from Embedding2D import Embedding2D


class RippleNet(tf.keras.Model):
    def __int__(self,args, ripple_set):
        super(RippleNet, self).__init__()
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.ripple_set = ripple_set
        self.ripple_size = args.ripple_size
        # 几跳
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = True
        # 先写死l2正则项
        self.l2 = tf.keras.regularizers.l2(args.l2)

    def call(self, inputs, training=None, mask=None):
        pass
        """
         self._build_embeddings()
        self._build_inputs()
        self._build_model()
        self._build_loss()
        print('self.user_id_shape',self.user_id.shape)
        model = Model(inputs=[self.user_id, self.item_id], outputs=self.score)
        model.add_loss(self.l2.l2 * self.l2_loss)
        model.add_loss(self.kge_weight * -self.kge_loss)
        return model
        """


    def ss(self, args, ripple_set):
        #super(RippleNet,self).__init__()
        #self._parse_args(args,ripple_set)
        self._build_embeddings()
        self._build_inputs()
        self._build_model()
        self._build_loss()
        print('self.user_id_shape',self.user_id.shape)
        model = Model(inputs=[self.user_id, self.item_id], outputs=self.score)
        model.add_loss(self.l2.l2 * self.l2_loss)
        model.add_loss(self.kge_weight * -self.kge_loss)
        return model

    def _parse_args(self, args, ripple_set):
        self.n_entity = args.n_entity
        self.n_relation = args.n_relation
        self.dim = args.dim
        self.ripple_set = ripple_set
        self.ripple_size = args.ripple_size
        #几跳
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = True
        #先写死l2正则项
        self.l2 = tf.keras.regularizers.l2(args.l2)

    def _build_embeddings(self):
        self.entity_embedding = tf.keras.layers.Embedding(self.n_entity, self.dim, embeddings_initializer='glorot_uniform',embeddings_regularizer=self.l2)
        #由于relation是要用来链接head和tail的，所以它的embedding的维度为dim * dim
        self.relation_embedding = Embedding2D(self.n_relation, self.dim, self.dim, embeddings_initializer='glorot_uniform',embeddings_regularizer=self.l2)


    def _build_inputs(self):
        #因为tf2没有占位符
        #self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        #self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name="labels")
        pass


    def _update_item(self,i, o):
        if self.item_update_mode == 'replace':
            i = o
        elif self.item_update_mode == 'plus':
            i = i + o
        elif self.item_update_mode == 'replace_transform':
            #相当于tf.matmul(o, self.transform_matrix)
            i = self.transform_matrix(o)
        elif self.item_update_mode == 'plus_transform':
            #相当于tf.matmul((i + o), self.transform_matrix)
            i = self.transform_matrix(i + o)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return i


    def _build_model(self):
        # 转换矩阵:在每跳末尾，更新item embeddings嵌入
        self.transform_matrix = tf.keras.layers.Dense(self.dim, use_bias=False, kernel_initializer='glorot_uniform',kernel_regularizer=self.l2)

        self.item_id = tf.keras.Input(shape=(), name='item_id', dtype=tf.int32)
        #entity_embedding中item_id对应的embedding
        self.item_embeddings = self.entity_embedding(self.item_id)
        self.user_id = tf.keras.Input(shape=(), name='user_id', dtype=tf.int32)

        #取出user_id对应的ripple_set
        self.ripple_sets = tf.gather(self.ripple_set, self.user_id)
        #self.ripple_sets：Tensor("GatherV2:0", shape=(None, 2, 3, 32), dtype=int32)
        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []

        for hop in range(self.n_hop):
            self.h_emb_list.append(self.entity_embedding(self.ripple_sets[:, hop, 0]))
            self.r_emb_list.append(self.relation_embedding(self.ripple_sets[:, hop, 1]))
            self.t_emb_list.append(self.entity_embedding(self.ripple_sets[:, hop, 2]))

        o_list = self._key_addressing()
        self.score = tf.squeeze(self.predict(self.item_embeddings, o_list))



    def _key_addressing(self):
        #每个hop的tail加权平均，权重是计算得到的相关度，相当于绿框。
        o_list = []
        for hop in range(self.n_hop):
            #取出每一波纹的head embedding，并扩增维度
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)
            #取出每一波纹的relation embedding，并乘head embedding
            #tf.squeeze所有维度为1的那些维都删掉
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)
            v = tf.expand_dims(self.item_embeddings, axis=2)
            #Rh, v相乘且激活后得到每个tail在其relation下的相关度probs_expanded
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)
            probs_normalized = tf.keras.activations.softmax(probs)
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)
            #tail_embedding乘对应相关度,tf.reduce_sum(axis=1)按行求和
            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)  # batch, dim
            self.item_embeddings = self._update_item(self.item_embeddings, o)
            o_list.append(o)
        return o_list

        #在得到olist之后，可以只用olist里面最后一个向量，也可相加所有的向量，来代表user-embedding，并最终计算得到预测值
    def predict(self, item_embeddings, o_list):
        user_embeddings = sum(o_list) if self   .using_all_hops else o_list[-1]
        score = tf.keras.layers.Activation('sigmoid', name='score')(tf.reduce_sum(item_embeddings * user_embeddings, axis=1))

        return score

    def _build_loss(self):
        # 知识图谱嵌入损失项
        self.kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)  # batch, ripple_size, 1, dim
            t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)  # batch, ripple_size, dim, 1
            hRt = tf.squeeze(h_expanded @ self.r_emb_list[hop] @ t_expanded)  # batch, ripple_size
            self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))

        # 额外的l2正则项，对RippleNet在ml1m数据集上的auc指标有明显提升
        self.l2_loss = 0
        for hop in range(self.n_hop):
            self.l2_loss += tf.reduce_sum(tf.square(self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_sum(tf.square(self.r_emb_list[hop]))
            self.l2_loss += tf.reduce_sum(tf.square(self.t_emb_list[hop]))
            if self.item_update_mode == "replace nonlinear" or self.item_update_mode == "plus nonlinear":
                self.l2_loss += tf.nn.l2_loss(self.transform_matrix)




























