# *_*coding:utf-8 *_*
from Recommender_System.algorithm.RippleNet.tool import get_user_positive_item_list, construct_directed_kg,get_ripple_set
from try_ripplenet import RippleNet
from Recommender_System.algorithm.RippleNet.train import train
import multiprocessing
import os
from Recommender_System.data import kg_loader, data_process
import tensorflow as tf
import argparse
from MKR import data_loader_MKR,preprocess_MKR
from Recommender_System.data.data_process import prepare_topk
#n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.ml1m_kg1m, negative_sample_threshold=4, split_ensure_positive=True)


n_user, n_item, n_entity, n_relation, train_data, test_data, kg = data_loader_MKR.load_data()


topk_sample_user= 100
topk_data = prepare_topk(train_data, test_data, n_user, n_item, topk_sample_user)


parser = argparse.ArgumentParser()

parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_relation', type=int, default=n_relation, help='n_relation')
parser.add_argument('--kge_weight', type=float, default=0.01, help='kge_weight')
parser.add_argument('--l2', type=float, default=1e-7, help='l2')
parser.add_argument('--n_entity', type=int, default=n_entity, help='n_entity')
parser.add_argument('--n_hop', type=int, default=2, help='n_hop')
parser.add_argument('--ripple_size', type=int, default=32, help='ripple_size')
parser.add_argument('--item_update_mode', type=str, default='plus_transform', help='item_update_mode')
#parser.add_argument('--use_all_hops', type=int, default=1, help='use_all_hops')

args = parser.parse_args()


#根据知识图谱结构构建有向图
directed_kg = construct_directed_kg(kg)
#得到每个用户在训练集上的正反馈物品集合
user_positive_item_list = get_user_positive_item_list(train_data)
#考虑两跳，每跳随机32个ripple set
n_hop, ripple_size =2,32
#根据知识图谱有向图得到每个用户每跳的三元组，', ('n_user', 'hop_size', 'ripple_size'
ripple_set = get_ripple_set(n_user, n_hop, ripple_size, user_positive_item_list, directed_kg)
#import pickle
#with open('/Users/wjj/Desktop/Nipple_GPU/big_ripple_set_list.pk', 'wb') as f:
    #pickle.dump(ripple_set,f)
#删除ripple_set中为空的用户
"""
index_list = []
for i in range(len(ripple_set)):
    if len(ripple_set[i]) == 0:
        index_list.append(i)
for i in sorted(index_list, reverse=True):
    del ripple_set[i]
n_user -= len(index_list)
print('n_user',n_user)
import pickle
with open('/Users/wjj/Desktop/Nipple_GPU/little_ripple_set_list.pk', 'wb') as f:
    pickle.dump(ripple_set,f)

"""


ripplenet = RippleNet()

model = ripplenet.ss(args,ripple_set)
train(model, train_data, test_data, topk_data, optimizer=tf.keras.optimizers.Adam(0.01), epochs=2, batch=512)


