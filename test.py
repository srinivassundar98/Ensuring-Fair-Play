from aif360.datasets import MEPSDataset19
dataset = MEPSDataset19()
df_meps, _ = dataset.convert_to_dataframe()
df_meps.to_csv('MEPSDataset19.csv')