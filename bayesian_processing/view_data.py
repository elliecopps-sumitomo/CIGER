import pandas as pd
import h5py
import cmapPy.pandasGEXpress.parse as parse

#cmap_data = parse.parse('./GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx')
#metadata = cmap_data.col_metadata_df

#print(metadata)
#filePath = "./Bayesian_GSE70138_Level5_COMPZ_n116218x978.h5?download=1"
filePath = "./Bayesian_GSE92742_Level5_COMPZ_n361481x978.h5?download=1"

with h5py.File(filePath, 'r') as file:
  #print("keys: ", list(file.keys()))
  #print(file["colid"][:20])
  #print(file["rowid"][:20])
  #print(file["data"][:20])  
  ids = file["colid"]
  genes = list(file["rowid"])
  debc_ids = set(["REP.A007_A375_24H:N13", "CPC006_A549_24H:BRD-K70792160-003-02-0:24", "REP.A007_HA1E_24H:N13", "CPC006_HCC515_6H:BRD-K70792160-003-02-0:24", "REP.A007_HELA_24H:N13", "REP.A007_HT29_24H:N13", "REP.A007_MCF7_24H:N13", "CPC006_PC3_24H:BRD-K70792160-003-02-0:24", "CPC006_VCAP_24H:BRD-K70792160-003-02-0:24", "REP.A007_YAPC_24H:N13"])
  df = pd.DataFrame(columns=["signature", "Name_GeneSymbol", "z_scores"])
  for i, drug in enumerate(ids):
    if drug in debc_ids:
      drug_data = {"signature": [drug] * 978, "Name_GeneSymbol":genes, "z_scores":file["data"][i]}
      drug_df = pd.DataFrame(drug_data)
      df = df.append(drug_df, ignore_index=True)
df.to_csv("GSE92742_10debc.csv", index=False)
  #def explore_group(group, indent=" "):
  #  for key in group:
  #    item = group[key]
  #    if isinstance(item, h5py.Group):
  #      print(indent, "Group: ",  key)
  #      explore_group(item, indent + " ")
  #    elif isinstance(item, h5py.Dataset):
  #      print(indent, "Dataset: ", key)
  #      print(indent, "Shape: ", item.shape)
  #      print(indent, "Data Type: ", item.dtype)
  #      for attr_name, attr_value in item.attrs.items():
  #        print(indent, "Attribute: ", attr_name, " = ", attr_value)
  #explore_group(file)
