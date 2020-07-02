# *_*coding:utf-8 *_*
from utils import *
from walk import *
from model import GATNEModel
import tqdm
import tensorflow as tf

def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size
    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
            neigh.append(neighbors[pairs[index][0]])
        yield (np.array(x).astype(np.int32), np.array(y).reshape(-1, 1).astype(np.int32), np.array(t).astype(np.int32), np.array(neigh).astype(np.int32))

def loss_fun(inputs, targets,weights,biases,num_sampled,num_nodes):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=weights, biases=biases, labels=targets,
            inputs=inputs, num_sampled=num_sampled, num_classes=num_nodes))

    return loss

def grad(model, inputs, targets,weights,biases,num_sampled,num_nodes):
    with tf.GradientTape() as tape:
        loss_value = loss_fun(inputs, targets,weights,biases,num_sampled,num_nodes)
    return tape.gradient(loss_value, model.trainable_variables)


def train_model(network_data,feature_dic,neighbor_samples):
    all_walks = generate_walks(network_data, args.num_walks, args.walk_length, args.schema, file_name)
    vocab, index2word = generate_vocab(all_walks)
    train_pairs = generate_pairs(all_walks, vocab, args.window_size)

    edge_types = list(network_data.keys())

    num_nodes = len(index2word)
    edge_type_count = len(edge_types)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_sampled = args.negative_samples
    dim_a = args.att_dim
    neighbor_samples = args.neighbor_samples

    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        g = network_data[edge_types[r]]
        # 把邻居节点放入list
        for (x, y) in g:
            ix = vocab[x].index
            iy = vocab[y].index
            neighbors[ix][r].append(iy)
            neighbors[iy][r].append(ix)
        # 对邻居节点采样
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:
                neighbors[i][r].extend(
                    list(np.random.choice(neighbors[i][r], size=neighbor_samples - len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:
                neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))

    GATNEmodel = GATNEModel(num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a,feature_dic,vocab,num_sampled,neighbor_samples)
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)

    g_iter = 0
    best_score = 0
    patience = 0
    for epoch in range(epochs):
        random.shuffle(train_pairs)
        batches = get_batches(train_pairs, neighbors, batch_size)

        data_iter = tqdm.tqdm(
            batches,
            desc="epoch %d" % (epoch),
            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
            bar_format="{l_bar}{r_bar}",
        )
        avg_loss = 0.0

        for i, data in enumerate(data_iter):
            last_node_embed,loss,opt_tmp = GATNEmodel(data[0], data[1], data[2], data[3])
            opt.minimize(loss_fun, var_list=GATNEmodel.trainable_variables,model = GATNEmodel,last_node_embed =last_node_embed)


            #grads = tape.gradient(loss, GATNEmodel.trainable_variables)
            #opt.apply_gradients(zip(grads, GATNEmodel.trainable_variables))

            avg_loss += loss
            g_iter += 1
            if i % 5000 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss,
                }
                data_iter.write(str(post_fix))
            print('1')

        final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
        print('2')
        #遍历每种边类型中的样本，获得embeding存入final_model
        for i in range(edge_type_count):
            for j in range(num_nodes):
                print('3')
                final_model[edge_types[i]][index2word[j]] = np.array(
                    GATNEmodel([j], [i], neighbors[j])[0])
        valid_aucs, valid_f1s, valid_prs = [], [], []
        test_aucs, test_f1s, test_prs = [], [], []

        for i in range(edge_type_count):
            if args.eval_type == "all" or edge_types[i] in args.eval_type.split(","):
                tmp_auc, tmp_f1, tmp_pr = evaluate(
                    final_model[edge_types[i]],
                    valid_true_data_by_edge[edge_types[i]],
                    valid_false_data_by_edge[edge_types[i]],
                )
                valid_aucs.append(tmp_auc)
                valid_f1s.append(tmp_f1)
                valid_prs.append(tmp_pr)

                tmp_auc, tmp_f1, tmp_pr = evaluate(
                    final_model[edge_types[i]],
                    testing_true_data_by_edge[edge_types[i]],
                    testing_false_data_by_edge[edge_types[i]],
                )
                test_aucs.append(tmp_auc)
                test_f1s.append(tmp_f1)
                test_prs.append(tmp_pr)
        print("valid auc:", np.mean(valid_aucs))
        print("valid pr:", np.mean(valid_prs))
        print("valid f1:", np.mean(valid_f1s))

        average_auc = np.mean(test_aucs)
        average_f1 = np.mean(test_f1s)
        average_pr = np.mean(test_prs)

        cur_score = np.mean(valid_aucs)
        if cur_score > best_score:
            best_score = cur_score
            patience = 0
        else:
            patience += 1
            if patience > args.patience:
                print("Early Stopping")
                break
    return average_auc, average_f1, average_pr



if __name__ == "__main__":

    args = parse_args()
    file_name = args.input
    print(args)

    if args.features is not None:
        feature_dic = {}
        with open('/Users/wjj/Desktop/GATNE/'+args.features, 'r') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                items = line.strip().split()
                feature_dic[items[0]] = items[1:]
    else:
        feature_dic = None

    #de = Child()


    training_data_by_type = load_training_data(file_name + "/train.txt")
    #valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + "/valid.txt")
    #testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + "/test.txt")

    average_auc, average_f1, average_pr = train_model(training_data_by_type,feature_dic,args.neighbor_samples)

    #print("Overall ROC-AUC:", average_auc)
    #print("Overall PR-AUC", average_pr)
    #print("Overall F1:", average_f1)