import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, ndcg_score, average_precision_score
from scipy.stats import pearsonr, spearmanr, kendalltau


def precision_k(label_test, label_predict, k):
    num_pos = 2 * k
    num_neg = 2 * k
    label_test = np.argsort(label_test, axis=1)
    label_predict = np.argsort(label_predict, axis=1)
    precision_k_neg = []
    precision_k_pos = []
    neg_test_set = label_test[:, :num_neg]
    pos_test_set = label_test[:, -num_pos:]
    neg_predict_set = label_predict[:, :k]
    pos_predict_set = label_predict[:, -k:] 
    for i in range(len(neg_test_set)):
        neg_test = set(neg_test_set[i])
        pos_test = set(pos_test_set[i])
        neg_predict = set(neg_predict_set[i])
        pos_predict = set(pos_predict_set[i])
        precision_k_neg.append(len(neg_test.intersection(neg_predict)) / k)
        precision_k_pos.append(len(pos_test.intersection(pos_predict)) / k)
    return np.mean(precision_k_pos)


def precision_k_200(label_test, label_predict):
    #OUTPUT: currently the average of the positive and negative precision scores
    #label_test is what we want to get, the pancreatic cancer signature, where the genes are in the same order as our prediction genes (1, 978)
    #label_predict has one row for each prediction (#drugs*10, 978)
    num_pos=200
    k = 200
    precision_k_pos = list([] for i in range(0, 10))
    precision_k_neg = list([] for i in range(0, 10))
    precision_k_tot = list([] for i in range(0, 10))

    #rank genes in terms of how much their expression changed
    label_test = np.argsort(label_test)
    label_predict = np.argsort(label_predict, axis=1)

    #Take only the k most up and down regulated genes
    neg_test_set = label_test[:num_pos]
    pos_test_set = label_test[-num_pos:]
    neg_predict_set = label_predict[:, :k]
    pos_predict_set = label_predict[:, -k:]
    pos_test = set(pos_test_set)
    neg_test = set(neg_test_set)
    for i in range(len(neg_predict_set)):
        neg_predict = set(neg_predict_set[i])
        pos_predict = set(pos_predict_set[i])
        precision_k_neg[i%10].append(len(neg_test.intersection(neg_predict)) / k)
        precision_k_pos[i % 10].append(len(pos_test.intersection(pos_predict)) / k)
        precision_k_tot[i%10].append((len(pos_test.intersection(pos_predict)) + len(neg_test.intersection(neg_predict)))/ k)
    #Take average of positive and negative scores
    # precision_scores = []
    # for i in range(0, 10):
    #     avg_lst = [(g+h) / 2 for g, h in zip(precision_k_pos[i], precision_k_neg[i])]
    #     precision_scores.append(avg_lst)
    # print("dimensions of scores:")
    # print(np.shape(precision_scores))
    return precision_k_tot, precision_k_pos, precision_k_neg #this is a 2d array where each cell line has a score for each drug tested



def kendall_tau(label_test, label_predict):
    score = []
    for lb_test, lb_predict in zip(label_test, label_predict):
        tau, p_value = kendalltau(lb_test, lb_predict)
        score.append(tau)
    return np.mean(score)


def mean_average_precision(label_test, label_predict):
    k = 200
    score = []
    pos_idx = np.argsort(label_test, axis=1)[:, (-k):]
    label_test_binary = np.zeros_like(label_test)
    for i in range(len(label_test_binary)):
        label_test_binary[i][pos_idx[i]] = 1
        score.append(average_precision_score(label_test_binary[i], label_predict[i]))
    return np.mean(score)


def rmse(label_test, label_predict):
    return np.sqrt(mean_squared_error(label_test, label_predict))


def correlation(label_test, label_predict, correlation_type):
    if correlation_type == 'pearson':
        corr = pearsonr
    elif correlation_type == 'spearman':
        corr = spearmanr
    else:
        raise ValueError("Unknown correlation type: %s" % correlation_type)
    score = []
    for lb_test, lb_predict in zip(label_test, label_predict):
        score.append(corr(lb_test, lb_predict)[0])
    return np.mean(score), score


def auroc(label_test, label_predict):
    label_test = label_test.reshape(-1)
    label_predict = label_predict.reshape(-1)
    return roc_auc_score(label_test, label_predict)


def auprc(label_test, label_predict):
    label_test = label_test.reshape(-1)
    label_predict = label_predict.reshape(-1)
    return average_precision_score(label_test, label_predict)


def ndcg(label_test, label_predict):
    return ndcg_score(label_test, label_predict)

def ndcg_k(label_test, label_predict, k):
    return ndcg_score(label_test, label_predict, k=k)

def ndcg_per_sample(label_test, label_predict, num_genes):
    score = []
    for i in range(len(label_test)):
        score.append(ndcg_score(label_test[i].reshape(1, num_genes), label_predict[i].reshape(1, num_genes)))
    return score


def ndcg_random(label_test):
    label_test = np.repeat(label_test, 100, axis=0)
    label_predict = np.array([np.random.permutation(978) for i in range(len(label_test))])
    return ndcg_score(label_test, label_predict)


def auroc_per_cell(label_test, label_predict, cell_idx):
    score = []
    for c_idx in cell_idx:
        lb_test = label_test[c_idx].reshape(-1)
        lb_predict = label_predict[c_idx].reshape(-1)
        score.append(roc_auc_score(lb_test, lb_predict))
    return score


def ndcg_per_cell(label_test, label_predict, cell_idx):
    score = []
    for c_idx in cell_idx:
        lb_test = label_test[c_idx]
        lb_predict = label_predict[c_idx]
        score.append(ndcg_score(lb_test, lb_predict))
    return score
