import cmapPy.pandasGEXpress.parse as parse

cmap_data = parse.parse('./LINCS_data_70138_unprocessed.gctx')
metadata = cmap_data.col_metadata_df

print(metadata)