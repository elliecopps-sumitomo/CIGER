import pandas as pd
import csv
import ast

#precision_top_x tells you how many elements in the top *pred_subset* of the prediction are in the top *actual_subset* of the actual value
#it's intended to be a slightly more forgiving version of P@k, so we can see how close we are to correctly predicting the top 10-20 differentially expressed genes
#parameters: absolute = take all top differentially expressed genes, positive and negative, negative = only look at down-regulated when true
def precision_top_x(pred_subset, actual_subset, prediction, actual, absolute=True, negative =False):
    if absolute:
        prediction = [abs(x) for x in prediction]
        actual = [abs(x) for x in actual]
    elif negative:
        prediction = [-x for x in prediction]
        actual = [-x for x in actual]


    #get list of the indexes of the top z-scores in both prediction and actual
    top_pred_genes = sorted(range(len(prediction)), key=lambda i: prediction[i])[-pred_subset:]
    top_actual_genes = set(sorted(range(len(actual)), key=lambda i: actual[i])[-actual_subset:])

    return len(set(top_pred_genes).intersection(top_actual_genes)) / pred_subset

def direction_of_change(prediction, actual, size):
    if size == 0:
        correct = 0
        for i in range(0, len(prediction)):
            if prediction[i] * actual[i] > 0:
                correct += 1
        return correct / len(prediction)
    #return the fraction of genes in the top 'size' of differentially expressed genes whose direction of change is correct

    #get indices of most up and down regulated genes
    pos_index = sorted(range(len(prediction)), key=lambda i: prediction[i])[-size:]
    neg_index = sorted(range(len(prediction)), key=lambda i: prediction[i])[:size]

    #check whether the most up and down regulated genes are predicted in the correct direction
    correct = 0
    for i in range(0, size):
        if actual[pos_index[i]] > 0:
            correct +=1
        if actual[neg_index[i]] < 0:
            correct +=1
    return correct/(size*2)


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
    debc_predictions = list(reader) #list of all 10 predictions, 1 for each cell line

prediction_precision = []
dox_precision = []
for i in range(0, 10):
    print(cell_lines[i])
    #make dataframe where the index is the genes, and the column has the z score for each
    actual = actual_df[(actual_df.signature == lincs_id_debc[i])]
    actual = actual[["Name_GeneSymbol", "z_scores"]]
    actual = actual.sort_values(by = ["Name_GeneSymbol"])
    actual = actual.set_index("Name_GeneSymbol")
    actual_list = actual['z_scores']

    pred = ast.literal_eval(debc_predictions[i][2])
    prediction = pd.DataFrame.from_dict(pred, orient='index')
    prediction = prediction.sort_index()
    prediction_list = list(prediction[0])

    if(lincs_id_doxorubicin[i] == 'skip'):
        dox_precision.append('NA')
    else:
        dox = dox_df[(dox_df.signatureID == lincs_id_doxorubicin[i])]
        dox = dox[["Name_GeneSymbol", "Value_LogDiffExp"]]
        dox = dox.sort_values(by = ["Name_GeneSymbol"])
        dox.columns = ["Name_GeneSymbol", "doxorubicin"]
        dox_list = dox["doxorubicin"]
        dox_precision.append(precision_top_x(10, 20, dox_list, actual_list))
        print("dox: ", direction_of_change(list(dox_list), actual_list, 0))

    

    prediction_precision.append(precision_top_x(10, 10, prediction_list, actual_list))
    print("pred: ", direction_of_change(prediction_list, actual_list, 0))

#print(dox_precision)
#print(prediction_precision)
#precision_df = pd.DataFrame({'prediction':prediction_precision, 'doxorubicin':dox_precision})
#precision_df.to_csv('precision_data/precision10in10.csv', index=False)
direction_of_change([1,2,3,4,5,6,7,8], [1,2,3,4,5,-6,-7,-8], 3)