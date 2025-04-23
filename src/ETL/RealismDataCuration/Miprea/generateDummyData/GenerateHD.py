import pandas as pd

sourceLocation = '/home/pliu/Downloads/MipREA_DATA/hervv3_mip.csv'
df = pd.read_csv(sourceLocation, sep=',')
#samples = df.head(100)
#samples.to_csv('hervv3_mip_samples.csv', index=False, header=True, encoding='utf-8')
totalRows = len(df.index)
totalColumns = len(df.columns)

print("total Rows: "+str(totalRows))
print("Total Columns: "+str(totalColumns))