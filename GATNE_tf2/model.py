# *_*coding:utf-8 *_*
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from numpy import random
from utils import *



class GATNEModel(Model):
    #self.feature_dic使用GATNE-I还是T，是否加上节点特征
    def __init__(self, num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a,feature_dic,vocab,num_sampled,neighbor_samples):
        super(GATNEModel,self).__init__()

        self.num_nodes = num_nodes
        print('self.num_nodes',self.num_nodes)
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_type_count = edge_type_count
        self.u_num = edge_type_count
        self.dim_a = dim_a
        #暂时写死
        self.att_head = 1
        self.vocab = vocab
        self.num_sampled = num_sampled
        self.feature_dic = feature_dic
        self.neighbor_samples = neighbor_samples
        self.node_embeddings = tf.Variable(tf.random.uniform([num_nodes, embedding_size], -1.0, 1.0))
        self.node_type_embeddings = tf.Variable(tf.random.uniform([num_nodes, self.edge_type_count, embedding_u_size], -1.0, 1.0))
        self.trans_weights = tf.Variable(tf.random.truncated_normal([edge_type_count, embedding_u_size, embedding_size // self.att_head], stddev=1.0 / math.sqrt(embedding_size)))
        self.trans_weights_s1 = tf.Variable(tf.random.truncated_normal([edge_type_count, embedding_u_size, dim_a], stddev=1.0 / math.sqrt(embedding_size)))
        self.trans_weights_s2 = tf.Variable(tf.random.truncated_normal([edge_type_count, dim_a, self.att_head], stddev=1.0 / math.sqrt(embedding_size)))
        self.nce_weights = tf.Variable(tf.random.truncated_normal([num_nodes, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        self.nce_biases = tf.Variable(tf.zeros([num_nodes]))

        if self.feature_dic is not None:

            self.feature_dim = len(list(self.feature_dic.values())[0])
            print('feature dimension: ' + str(self.feature_dim))
            self.features = np.zeros((self.num_nodes, self.feature_dim), dtype=np.float32)
            for key, value in self.feature_dic.items():
                if key in self.vocab:
                    self.features[self.vocab[key].index, :] = np.array(value)

            self.node_features = tf.Variable(self.features, name='node_features', trainable=False)
            self.feature_weights = tf.Variable(tf.random.truncated_normal([self.feature_dim, self.embedding_size], stddev=1.0))
            self.linear = tf.keras.layers.Dense(units=self.embedding_size, activation=tf.nn.tanh, use_bias=True)

            self.embed_trans = tf.Variable(tf.random.truncated_normal([self.feature_dim, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
            #u_embed_trans似乎是边的类型相结合的embed
            self.u_embed_trans = tf.Variable(tf.random.truncated_normal([self.edge_type_count, self.feature_dim, self.embedding_u_size], stddev=1.0 / math.sqrt(self.embedding_size)))


    def call(self,train_inputs, train_labels,train_types, node_neigh):

        train_labels_tensor = tf.convert_to_tensor(train_labels)
        self.train_labels = tf.cast(train_labels_tensor, tf.int64)

        if self.feature_dic is not None:
            node_embed = tf.nn.embedding_lookup(self.node_features, train_inputs)
            node_embed = tf.matmul(node_embed, self.embed_trans)
        else:
            node_embed = tf.nn.embedding_lookup(self.node_embeddings, train_inputs)


        if self.feature_dic is not None:
            node_embed_neighbors = tf.nn.embedding_lookup(self.node_features, node_neigh)
            # 四维的向量[训练集中节点数量，边的类型数量，固定的序列长度，最初的节点特征维度]
            #把第k阶邻居，不同类型的边节点，对应的edge embedding进行concat聚合：
            node_embed_tmp = tf.concat([tf.matmul(
                tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, 0], [-1, 1, -1, -1]), [-1, self.feature_dim]),
                tf.reshape(tf.slice(self.u_embed_trans, [i, 0, 0], [1, -1, -1]), [self.feature_dim, self.embedding_u_size])
            )for i in range(self.edge_type_count)],axis=0)
            #tf.transpose转置，tf.reduce_mean求平均
            #将node_embed_tmp求平均就是node_type_embed
            node_type_embed = tf.transpose(tf.reduce_mean(tf.reshape(node_embed_tmp, [self.edge_type_count, -1, self.neighbor_samples, self.embedding_u_size]), axis=2), perm=[1,0,2])
        else:
            node_embed_neighbors = tf.nn.embedding_lookup(self.node_type_embeddings, node_neigh)
            node_embed_tmp = tf.concat([tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, i, 0], [-1, 1, -1, 1, -1]), [1, -1, self.neighbor_samples, self.embedding_u_size]) for i in range(self.edge_type_count)], axis=0)
            node_type_embed = tf.transpose(tf.reduce_mean(node_embed_tmp, axis=2), perm=[1,0,2])

        trans_w = tf.nn.embedding_lookup(self.trans_weights, train_types)
        trans_w_s1 = tf.nn.embedding_lookup(self.trans_weights_s1, train_types)
        trans_w_s2 = tf.nn.embedding_lookup(self.trans_weights_s2, train_types)

        attention = tf.reshape(tf.nn.softmax(
            tf.reshape(tf.matmul(tf.tanh(tf.matmul(node_type_embed, trans_w_s1)), trans_w_s2), [-1, self.u_num])),
                               [-1, self.att_head, self.u_num])
        node_type_embed = tf.matmul(attention, node_type_embed)
        node_embed = node_embed + tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, self.embedding_size])

        if self.feature_dic is not None:
            node_feat = tf.nn.embedding_lookup(self.node_features, train_inputs)
            node_embed = node_embed + tf.matmul(node_feat, self.feature_weights)

        last_node_embed = tf.nn.l2_normalize(node_embed, axis=1)
        #tf.nn.nec_loss()在skip_gram中使用负采样的函数，计算每个batch数据的损失函数
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights, biases=self.nce_biases, labels=self.train_labels, inputs=last_node_embed,num_sampled=self.num_sampled, num_classes=self.num_nodes))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)


        return last_node_embed,loss,optimizer















