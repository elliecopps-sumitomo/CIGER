CUDA_VISIBLE_DEVICES=0 python3 train.py --drug_file "../kaggle_data/drug_smiles_kaggle.csv" \
--drug_id_file "../kaggle_data/drug_id_kaggle.csv" --gene_file "../kaggle_data/gene_feature_kaggle.csv" \
--data_file "../kaggle_data/chemical_signature_kaggle.csv" --fp_type 'neural' --label_type 'real' --loss_type 'list_wise_rankcosine' \
--batch_size 64 --max_epoch 100 --lr 0.003 --fold 0 --model_name 'kaggle_100epoch_lr003' --warm_start 'False'  --inference 'False' \
--kaggle 'True' > 'output/kaggle_100.txt'
