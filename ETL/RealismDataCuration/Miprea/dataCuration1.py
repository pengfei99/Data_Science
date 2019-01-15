#In This file we use MipRea_global.xslx to build miprea_viral_reactivation.csv
import pandas as pd
from utils import renameColumn

Location = '/home/pliu/Downloads/MipREA_DATA/batch_MipREA_global.xlsx'
df = pd.read_excel(Location,0)

sampleIDs = []
sampleTimePoints = []

def parseTimePoint(kinetic):
    if kinetic == "J1":
        return 1
    elif kinetic == "J3/4":
        return 2
    elif kinetic == "J5/7":
        return 3
    else:
        return 0


def parseSampleID(patientID, kinetic):
    if kinetic == "J1":
        return patientID+"_"+"D1"
    elif kinetic == "J3/4":
        return patientID+"_"+"D3"
    elif kinetic == "J5/7":
        return patientID+"_"+"D6"
    else:
        return "BAD_D9"

#calculate sampleID and time point
for index, row in df.iterrows():
    patientID = str(row['Patient_ID'])
    kinetic = str(row['kinetic'])
    timePoint = parseTimePoint(kinetic)
    sampleID = parseSampleID(patientID, kinetic)
    sampleIDs.append(sampleID)
    sampleTimePoints.append(timePoint)

# Build new data frame with Sample_ID and time_point
newSamples = pd.DataFrame.from_dict({'Sample_ID': sampleIDs,
                           'Time_point': sampleTimePoints})

# add sample_id and time_point to the original data frame
df = pd.concat([df, newSamples], axis=1)

# change column name base on the document
df = renameColumn(df, 'EBV (cp/mL)', 'EBV_logCpmL')
df = renameColumn(df, 'CMV (cp/mL)', 'CMV_logCpmL')
df = renameColumn(df, 'HSV1 (cp/mL)', 'HSV1_logCpmL')
df = renameColumn(df, 'HHV6 (cp/mL)', 'HHV6_logCpmL')
df = renameColumn(df, 'HHV7 (Ct)', 'HHV7_Ct')
df = renameColumn(df, 'TTV (cp/mL)', 'TTV_logCpmL')

#df.to_excel('miprea_viral_reactivation.xlsx', index=True, header=True)
#df.to_csv('miprea_viral_reactivation.csv', index=True, header=True, encoding='utf-8')

test = df.head(7)
print(test)
test.to_csv('test.csv', index=True, header=True, encoding='utf-8')
test.to_excel('test.xlsx',index=False, header=True)
