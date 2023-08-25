import csv
from metric import precision_k_200
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--disease_dseq', help='dseq2 profile of disease treatment you want to immitate')

args = parser.parse_args()
disease_dseq = args.disease_dseq

#make dictionary of genes, where gene name is the key and the gene's order/index is the value
reader = csv.reader(open('../data/gene_feature.csv'))
gene_file_list = list(reader)
genes = dict()
for ind, g in enumerate(gene_file_list):
    genes[g[0]] = ind
print(genes)
#Looping through expression data for disease, make an array of the expression values of our selected genes, in the order of the gene_feature file
expression_vals = [0] * len(genes)
data_file = csv.reader(open(disease_dseq))
for gene_exp in data_file:
    if(gene_exp[0]) in genes:
        expression_vals[genes[gene_exp[0]]] = (float(gene_exp[2]))
    elif not isinstance(gene_exp[2], str):
        expression_vals.append(float(gene_exp[2]))
print(expression_vals)
np.save('../disease_profile/pancreatic_expression_profile_all.npy', np.array(expression_vals))

