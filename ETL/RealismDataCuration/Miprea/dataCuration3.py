# This scipt transforme annotation_HERVV3.csv to hervv3_metadata.csv
import pandas as pd
from utils import renameColumn

Location = '/home/pliu/Downloads/MipREA_DATA/annotation_HERVV3.csv'
df = pd.read_csv(Location,sep=';' )
#names=['probesetName',REFSEQ;ALIAS;repertoire;functional_region;subregion_chr_location;element_chr_location;CHR;CHRLOC;CHRLOCEND;organisme;id_dfam;superFamily_dfam;simple_family]
samples = df.head()
print(samples)

samples = renameColumn(samples, 'probesetName', 'probeset_ID')
samples = renameColumn(samples, 'ALIAS', 'probbeset_Name')
print(samples)
samples.to_csv('hervv3_metadata.csv', index=False, header=True, encoding='utf-8')