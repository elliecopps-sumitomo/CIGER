import numpy as np
from collections import Counter


np.set_printoptions(threshold=np.inf)


def get_l1000_gene(input_file):
    with open(input_file, 'r') as f:
        output = f.readline().strip().split(',')
        return output


def read_disease_sig(input_file):
    disease_gene = []
    disease_sig = []
    with open(input_file, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            disease_gene.append(line[0].replace('"', ''))
            disease_sig.append(float(line[1]))
    return disease_gene, np.array(disease_sig)


def read_disease_sig1(input_file):
    disease_gene = []
    disease_sig = []
    with open(input_file, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            if 'NA' not in line:
                disease_gene.append(line[-1])
                disease_sig.append(float(line[2]))
    return disease_gene, np.array(disease_sig)


def read_drug_sig(input_file):
    drug_sig = np.load(input_file)
    return drug_sig


def calculate_precision(drug_idx, disease_idx):
    disease_idx = set(disease_idx)
    z = len(disease_idx)
    drug_shape = np.shape(drug_idx)
    score = np.zeros((drug_shape[0], drug_shape[1]))
    for i in range(drug_shape[0]):
        for j in range(drug_shape[1]):
            score[i, j] = len(set(drug_idx[i, j]).intersection(disease_idx)) / z
    return score


def read_drug_id(input_file):
    with open(input_file, 'r') as f:
        drug_id = f.readline().strip().split(',')
    return drug_id


#score = np.load('../scores/k_scores_pos.npy') #score where each row is array of score for each drug tested
score = np.load('precision_score.npy')
test = [i[0] for i in score]
top_10_drug_idx = np.argsort(score, axis=-1)[:, -10:] #Finds index of top 10 scores for each cell line, puts them into an array
#print(top_10_drug_idx)
#drug_id = read_drug_id('pancreatic_drug_id.csv') #list of drug ids IN ALPHABETIC ORDER
drug_id = read_drug_id('drugbank_drug_id.csv')
drug_list_10 = Counter(top_10_drug_idx.flatten()) #Turns into a dictionary of form drug_index : number of cell lines where it has top 10 score
#print(drug_list_10)
set10 = set(drug_list_10.keys())
#print(set10)
output = [['DrugBank ID', 'Number of cell lines of which drug appears in top-10 candidate']]
c1 = 'DrugBank ID'
c2 = 'Number of cell lines of which drug appears in top-10 candidate list'
print("{:<15} {:<15}".format(c1, c2))
# print(drug_list_10)
sorted_lst = sorted(drug_list_10.items(), key=lambda item: item[1], reverse=True)
for k, v in sorted_lst:
    print("{:<15} {:<15}".format(drug_id[k], str(v)))