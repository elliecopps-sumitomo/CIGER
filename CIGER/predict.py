import os

import torch
import numpy as np
import argparse
from datetime import datetime
import csv

from models import CIGER
from utils import DataReaderPredict, DataReader, precision_k_200
from tqdm import tqdm

start_time = datetime.now()

parser = argparse.ArgumentParser(description='Gene Expression Prediction')
parser.add_argument('--drug_smiles', help='drug feature file (ECFP or SMILES)')
parser.add_argument('--gene_file', help='gene feature file')
parser.add_argument('--prediction_file', help='csv file where you want to store the predictions from the model')
parser.add_argument('--gsea_save', help='optional boolean whether or not to store gsea files for drug repurposing analysis')

# parser.add_argument('--fp_type', help='ECFP or Neural FP')
# parser.add_argument('--loss_type', help='pair_wise_ranknet/list_wise_listnet/list_wise_listmle/list_wise_rankcosine/'
#                                         'list_wise_ndcg')
args = parser.parse_args()

#csv in format: drug_id, drug_smile <line-break>
#list all drugs to be checked
drug_smiles = args.drug_smiles


#file should be included already, has 978 genes
gene_file = args.gene_file
save_predictions = args.prediction_file
gsea_save = args.gsea_save

fp_type = 'neural'
loss_type = 'list_wise_rankcosine' 
label_type = 'real'
fold = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
intitializer = torch.nn.init.xavier_uniform_

reader = csv.reader(open(gene_file))
gene_file_list = list(reader)
num_gene = len(gene_file_list)
gene_list = []
for g in gene_file_list:
    gene_list.append(g[0])


data = DataReaderPredict(drug_smiles, gene_file, device)

model = CIGER(drug_input_dim=data.drug_dim, gene_embed=data.gene, gene_input_dim=data.gene.size()[1],
                encode_dim=512, fp_type=fp_type, loss_type=loss_type, label_type=label_type, device=device,
                initializer=intitializer, pert_type_input_dim=1, cell_id_input_dim=10, pert_idose_input_dim=1, use_cell_id=True, use_pert_idose=False, use_pert_type=False)
checkpoint = torch.load('saved_model/ciger/best_model_decrease_lr.ckpt',
                        map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

predict_np = np.empty([0, num_gene])
batch = data.get_batch_data(1)
predictions = list()
k_score_predictions = list()

gsea_col_1 = ["NAME"] + gene_list
gsea_col_2 = ["DESCRIPTION"] + ['na']*len(gene_list)
gsea_predictions = [gsea_col_1, gsea_col_2]
cell_lines = ['A375', 'A549', 'HA1E', 'HCC515', 'HELA', 'HT29', 'MCF7', 'PC3', 'VCAP', 'YAPC']

for batch in (tqdm(batch)):
    pert_type = None
    drug = batch["drug"]
    pert_idose = None
    gene = batch["gene"]
    predict_np = np.empty([0, num_gene])

    #test drug over all cell lines, 
    for cell_idx in range(0, 10):
        prediction = dict()
        id_mtx =  np.zeros(10, dtype=int)
        id_mtx[cell_idx] = 1
        cell_id = torch.tensor([id_mtx])
        predict = model(drug, gene, pert_type, cell_id, pert_idose)
        prediction["drug_id"] = batch['drug_id']
        prediction["cell_line"] = cell_lines[cell_idx]
        gene_vals = predict.detach().cpu().tolist()[0]
        gsea_prediction_list = [batch['drug_id']+'_'+cell_lines[cell_idx]] + gene_vals
        prediction["gene_exp"] = dict(zip(gene_list, gene_vals))
        predictions.append(prediction)
        k_score_predictions.append(gene_vals)
        gsea_predictions.append(gsea_prediction_list)

#pancreatic_expression = np.load('disease_profile/pancreatic_expression_profile_all.npy') #This is log2FoldChange for each gene in gene_feature
# k_score_avgs, k_pos, k_neg = precision_k_200(pancreatic_expression, np.array(k_score_predictions))
# print("tot:")
# print(k_score_avgs)
# print("pos:")
# print(k_pos)
# for i in range(0, 9):
#     print(np.argsort([k[i] for k in k_score_avgs]))
#     print("pos")
#     print(np.argsort([k[i] for k in k_pos]))
#     print("")

# np.save('scores/k_scores.npy', k_score_avgs)
# np.save('scores/k_scores_pos.npy', k_pos)
# np.save('scores/k_scores_neg.npy', k_neg)

with open(save_predictions, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=["drug_id", "cell_line", "gene_exp"])
    writer.writeheader()
    writer.writerows(predictions)

# if gsea_save:
#     pancreatic_data = ["p_treated"] + pancreatic_expression.tolist()
#     pancreatic_string = "\t".join(map(str, pancreatic_data))
#     for i in range(2, len(gsea_predictions)):
#         gsea_transposed = list(zip(gsea_predictions[0], gsea_predictions[1], gsea_predictions[i], pancreatic_data))
#         with open('data/gsea_expressions/' + gsea_predictions[i][0] + '.txt', 'w') as file:
#             for row in gsea_transposed:
#                 row_str = "\t".join(map(str, row))
#                 file.write(row_str + "\n")
