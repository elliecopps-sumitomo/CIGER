#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python3 train.py --drug_file "data/drug_smiles.csv" \
--drug_id_file "data/drug_id.csv" --gene_file "data/gene_feature.csv" \
--data_file "data/chemical_signature.csv" --fp_type 'neural' --label_type 'real' --loss_type 'list_wise_rankcosine' \
--batch_size 64 --max_epoch 120 --lr 0.003 --fold 0 --model_name 'ndcg@20_120_epochs' --warm_start 'False'  --inference 'False' > 'output/ndcg@20_120_epochs.txt'
