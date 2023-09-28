import ast
import csv
import pandas as pd
from bioinfokit import analys, visuz
import numpy as np
from scipy.stats import spearmanr

#actual_df = pd.read_csv('gene_values.csv',sep = '\t')
actual_1 = pd.read_csv('../../bayesian_processing/GSE70138_10debc.csv')
actual_2 = pd.read_csv('../../bayesian_processing/GSE92742_10debc.csv')
actual_df = pd.concat([actual_1, actual_2], ignore_index=True)

dox_df = pd.read_csv('doxorubicin_expression.csv', sep = '\t')

#lincs_id_debc = ['LINCSCP_73439', 'LINCSCP_4588', 'LINCSCP_91774', 'LINCSCP_19581', 'LINCSCP_103584', 'LINCSCP_120377', 'LINCSCP_136956', 'LINCSCP_45004', 'LINCSCP_56929', 'LINCSCP_168763']
lincs_id_debc = ["REP.A007_A375_24H:N13", "CPC006_A549_24H:BRD-K70792160-003-02-0:24", "REP.A007_HA1E_24H:N13", "CPC006_HCC515_6H:BRD-K70792160-003-02-0:24", "REP.A007_HELA_24H:N13", "REP.A007_HT29_24H:N13", "REP.A007_MCF7_24H:N13", "CPC006_PC3_24H:BRD-K70792160-003-02-0:24", "CPC006_VCAP_24H:BRD-K70792160-003-02-0:24", "REP.A007_YAPC_24H:N13"]
lincs_id_doxorubicin = ['LINCSCP_80192', 'LINCSCP_191833', 'LINCSCP_98535', 'skip', 'LINCSCP_110365', 'LINCSCP_127168', 'LINCSCP_143753', 'LINCSCP_161954', 'skip', 'LINCSCP_175544']
cell_lines = ['A375', 'A549', 'HA1E', 'HCC515', 'HELA', 'HT29', 'MCF7', 'PC3', 'VCAP', 'YAPC']

with open('../../CIGER/10debc_predictions_lr.csv') as f:
    reader = csv.reader(f)
    next(reader, None)
    debc_predictions = list(reader)

correlation_df = pd.DataFrame(columns=["pred_corr", "cell_line"])
#correlation_df = pd.DataFrame(columns=["pred_corr", "dox_corr", "cell_line"])

for i in range(0, 10):
    # if(lincs_id_doxorubicin[i] == 'skip'):
    #     continue

    actual = actual_df[(actual_df.signature == lincs_id_debc[i])]
    actual = actual[["Name_GeneSymbol", "z_scores"]]
    actual = actual.sort_values(by = ["Name_GeneSymbol"])
    actual = actual.set_index("Name_GeneSymbol")

    # dox = dox_df[(dox_df.signatureID == lincs_id_doxorubicin[i])]
    # dox = dox[["Name_GeneSymbol", "Value_LogDiffExp"]]
    # dox = dox.sort_values(by = ["Name_GeneSymbol"])
    # dox.columns = ["Name_GeneSymbol", "doxorubicin"]
    # dox = dox.set_index("Name_GeneSymbol")

    pred = ast.literal_eval(debc_predictions[i][2])
    prediction = pd.DataFrame.from_dict(pred, orient='index')
    prediction = prediction.sort_index()

    heatmap_df = prediction.join(actual["z_scores"])
    #heatmap_df = heatmap_df.join(dox["doxorubicin"])
    heatmap_df.columns = ["predicted", "actual"]
    #heatmap_df["predicted"] = np.log2(heatmap_df["predicted"])
    
    heatmap_df = heatmap_df.sort_values(by="predicted", key=abs)
    heatmap_df = heatmap_df.tail(100)
    heatmap_df["actual"].fillna(0, inplace=True)
    heatmap_df["predicted"].fillna(0, inplace=True)
    #heatmap_df["dox"].fillna(0, inplace=True)
    correlation, p_value = spearmanr(list(heatmap_df["predicted"]), list(heatmap_df["actual"]))
    #d_correlation, d_p_value = spearmanr(list(heatmap_df["dox"]), list(heatmap_df["actual"]))
    print("Cell line: ", cell_lines[i])
    print("My prediction correlation: ", correlation)
    print("p_val: ", p_value)
    #print("Doxorubicin correlation: ", d_correlation)
    #print("Dox p_value", d_p_value)
    #correlation_df = correlation_df.loc[len(correlation_df.index)] = [correlation, d_correlation, cell_lines[i]]
    
    correlation_df = correlation_df.append({"pred_corr":correlation, "cell_line": cell_lines[i]}, ignore_index=True)
    visuz.gene_exp.hmap(df=heatmap_df, rowclus=False, colclus=False, dim=(3,6), tickfont=(6,4), figname="lr_processed_" + cell_lines[i])
#correlation_df.to_csv('rank_correlation_lr.csv', index=False)