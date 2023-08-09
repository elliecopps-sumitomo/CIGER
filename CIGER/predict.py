import os

import torch
import numpy as np
import argparse
from datetime import datetime

from models import CIGER
from utils import DataReaderDB, DataReader
from tqdm import tqdm

start_time = datetime.now()

parser = argparse.ArgumentParser(description='Gene Expression Prediction')
parser.add_argument('--drug_smiles', help='drug feature file (ECFP or SMILES)')
parser.add_argument('--drug_id_file', help='drug id file')
parser.add_argument('--gene_file', help='gene feature file')
parser.add_argument('--data_file', help='chemical signature file')
parser.add_argument('--model_name', help='name of model')
parser.add_argument('--fp_type', help='ECFP or Neural FP')
parser.add_argument('--loss_type', help='pair_wise_ranknet/list_wise_listnet/list_wise_listmle/list_wise_rankcosine/'
                                        'list_wise_ndcg')
args = parser.parse_args()

#csv in format: drug_id, drug_smile <line-break>
drug_smiles = args.drug_smiles

#csv of all drug ids to be checked
drug_ids = "data/drug_id.csv"

#file should be included already, has 978 genes
gene_file = args.gene_file



chem_file = "data/chemical_signature.csv"
# model_name = args.model_name
fp_type = args.fp_type
loss_type = args.loss_type 
label_type = 'binary'
fold = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
intitializer = torch.nn.init.xavier_uniform_
num_gene = 978
data = DataReaderDB(drug_smiles, gene_file, device)
dataTest = DataReader("data/drug_smiles.csv", drug_ids, gene_file, chem_file, fp_type, device, fold)

print(dataTest.pert_idose_dim)
print(dataTest.pert_type_dim)
print(dataTest.cell_id_dim)
print(dataTest.use_pert_type)

model = CIGER(drug_input_dim=data.drug_dim, gene_embed=data.gene, gene_input_dim=data.gene.size()[1],
                encode_dim=512, fp_type=fp_type, loss_type=loss_type, label_type=label_type, device=device,
                initializer=intitializer, pert_type_input_dim=1, cell_id_input_dim=10, pert_idose_input_dim=1, use_cell_id=True, use_pert_idose=False, use_pert_type=False)
checkpoint = torch.load('saved_model/ciger/ciger_list_wise_rankcosine_binary_testing_0.ckpt',
                        map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# label_binary_np = np.empty([0, num_gene])
# label_real_np = np.empty([0, num_gene])
predict_np = np.empty([0, num_gene])
batch = data.get_batch_data(1, 2)
for batch in (tqdm(batch)):
    pert_type = None
    drug = batch["drug"]
    cell_id = 0
    pert_idose = None
    gene = batch["gene"]

    #change this to test all of the cell_id's
    cell_id = torch.tensor([[1,0,0,0,0,0,0,0,0,0]])

    predict = model(drug, gene, pert_type, cell_id, pert_idose)
    predict_np = np.concatenate((predict_np, predict.detach().cpu().numpy()), axis=0)
print(predict_np)
print(np.shape(predict_np))
