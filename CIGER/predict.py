import os

import torch
import numpy as np
import argparse
from datetime import datetime
import csv

from models import CIGER
from utils import DataReaderPredict, DataReader
from tqdm import tqdm

start_time = datetime.now()

parser = argparse.ArgumentParser(description='Gene Expression Prediction')
parser.add_argument('--drug_smiles', help='drug feature file (ECFP or SMILES)')
parser.add_argument('--drug_id_file', help='drug id file')
parser.add_argument('--gene_file', help='gene feature file')
parser.add_argument('--data_file', help='chemical signature file')
parser.add_argument('--model_name', help='name of model')
parser.add_argument('--prediction_file', help='csv file where you want to store the predictions from the model')
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

fp_type = 'neural'
loss_type = 'list_wise_rankcosine' 
label_type = 'binary'
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
checkpoint = torch.load('saved_model/ciger/ciger_list_wise_rankcosine_binary_testing_0.ckpt',
                        map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

predict_np = np.empty([0, num_gene])
batch = data.get_batch_data(1)
predictions = list()
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
        prediction["gene_exp"] = dict(zip(gene_list, predict.detach().cpu().tolist()[0]))
        predictions.append(prediction)
print(predictions)
with open(save_predictions, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=["drug_id", "cell_line", "gene_exp"])
    writer.writeheader()
    writer.writerows(predictions)
