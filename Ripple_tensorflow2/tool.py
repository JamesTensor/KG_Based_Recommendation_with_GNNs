# *_*coding:utf-8 *_*
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np
import tensorflow as tf
from typing import List, Callable, Dict
from dataclasses import dataclass
from typing import Tuple, List, Callable, Dict

#'得到每个用户在训练集上的正反馈物品集合'
def get_user_positive_item_list(train_data: List[Tuple[int, int, int]]) -> Dict[int, List[int]]:
    user_positive_item_list = defaultdict(list)
    for user_id, item_id, label in train_data:
        if label == 1:
            user_positive_item_list[user_id].append(item_id)
    return user_positive_item_list


#'根据知识图谱结构构建有向图'
def construct_directed_kg(kg: List[Tuple[int, int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    kg_dict = defaultdict(list)
    for head_id, relation_id, tail_id in kg:
        kg_dict[head_id].append((relation_id, tail_id))
    return kg_dict


#'根据知识图谱有向图得到每个用户每跳的三元组，', ('n_user', 'hop_size', 'ripple_size'))
def get_ripple_set(n_user: int, hop_size: int, ripple_size: int, user_positive_item_list: Dict[int, List[int]],
                   kg_dict: Dict[int, List[Tuple[int, int]]]) -> List[List[Tuple[List[int], List[int], List[int]]]]:
    ripple_set = [[] for _ in range(n_user)]  # user_id -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]

    for user_id, positive_item_list in user_positive_item_list.items():
        for hop in range(hop_size):
            ripple_h, ripple_r, ripple_t = [], [], []
            tails_of_last_hop = positive_item_list if hop == 0 else ripple_set[user_id][-1][2]

            for entity_id in tails_of_last_hop:
                for relation_id, tail_id in kg_dict[entity_id]:
                    ripple_h.append(entity_id)
                    ripple_r.append(relation_id)
                    ripple_t.append(tail_id)

            if len(ripple_h) == 0:  # 如果当前用户当前跳的实体关系集合是空的
                ripple_set[user_id].append(ripple_set[user_id][-1])  # 仅复制上一跳的集合
            else:  # 对当前跳随机采样固定大小的实体关系集合
                replace = len(ripple_h) < ripple_size
                indices = np.random.choice(len(ripple_h), size=ripple_size, replace=replace)
                ripple_h = [ripple_h[i] for i in indices]
                ripple_r = [ripple_r[i] for i in indices]
                ripple_t = [ripple_t[i] for i in indices]
                ripple_set[user_id].append((ripple_h, ripple_r, ripple_t))

    return ripple_set


def prepare_ds(train_data: List[Tuple[int, int, int]], test_data: List[Tuple[int, int, int]],
               batch: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    def xy(data):
        user_ids = tf.constant([d[0] for d in data], dtype=tf.int32)
        item_ids = tf.constant([d[1] for d in data], dtype=tf.int32)
        labels = tf.constant([d[2] for d in data], dtype=tf.keras.backend.floatx())
        return {'user_id': user_ids, 'item_id': item_ids}, labels

    train_ds = tf.data.Dataset.from_tensor_slices(xy(train_data)).shuffle(len(train_data)).batch(batch)
    test_ds = tf.data.Dataset.from_tensor_slices(xy(test_data)).batch(batch)

    return train_ds, test_ds

def get_score_fn(model):
    @tf.function(experimental_relax_shapes=True)
    def _fast_model(ui):
        return tf.squeeze(model(ui))

    def score_fn(ui):
        ui = {k: tf.constant(v, dtype=tf.int32) for k, v in ui.items()}
        return _fast_model(ui).numpy()

    return score_fn



def log(epoch, train_loss, train_auc, train_precision, train_recall, test_loss, test_auc, test_precision, test_recall):
    train_f1 = 2. * train_precision * train_recall / (train_precision + train_recall) if train_precision + train_recall else 0
    test_f1 = 2. * test_precision * test_recall / (test_precision + test_recall) if test_precision + test_recall else 0
    print('epoch=%d, train_loss=%.5f, train_auc=%.5f, train_f1=%.5f, test_loss=%.5f, test_auc=%.5f, test_f1=%.5f' %
          (epoch + 1, train_loss, train_auc, train_f1, test_loss, test_auc, test_f1))







@dataclass
class TopkData:
    test_user_item_set: dict  # 在测试集上每个用户可以参与推荐的物品集合
    test_user_positive_item_set: dict  # 在测试集上每个用户有行为的物品集合


@dataclass
class TopkStatistic:
    hit: int = 0  # 命中数
    ru: int = 0  # 推荐数
    tu: int = 0  # 行为数


def topk_evaluate(topk_data: TopkData, score_fn: Callable[[Dict[str, List[int]]], List[float]],
                  ks=[1, 2, 5, 10, 20, 50, 100]) -> Tuple[List[float], List[float]]:
    kv = {k: TopkStatistic() for k in ks}
    for user_id, item_set in topk_data.test_user_item_set.items():
        ui = {'user_id': [user_id] * len(item_set), 'item_id': list(item_set)}
        item_score_list = list(zip(item_set, score_fn(ui)))
        sorted_item_list = [x[0] for x in sorted(item_score_list, key=lambda x: x[1], reverse=True)]

        positive_set = topk_data.test_user_positive_item_set[user_id]
        for k in ks:
            topk_set = set(sorted_item_list[:k])
            kv[k].hit += len(topk_set & positive_set)
            kv[k].ru += len(topk_set)
            kv[k].tu += len(positive_set)
    return [kv[k].hit / kv[k].ru for k in ks], [kv[k].hit / kv[k].tu for k in ks]  # precision, recall

def topk(topk_data: TopkData, score_fn: Callable[[Dict[str, List[int]]], List[float]], ks=[10, 36, 100]):
    precisions, recalls = topk_evaluate(topk_data, score_fn, ks)
    for k, precision, recall in zip(ks, precisions, recalls):
        f1 = 2. * precision * recall / (precision + recall) if precision + recall else 0
        print('[k=%d, precision=%.3f%%, recall=%.3f%%, f1=%.3f%%]' %
              (k, 100. * precision, 100. * recall, 100. * f1), end='')
    print()