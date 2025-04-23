# -*- coding: utf-8 -*-

import pandas as pd
from transMartCuration1 import curateGlobal
from utils import fillEmptyStringCell
from utils import fillAllEmptyDigitalCell

####################################
####Curation of Miprea Global############
####################################

# #Specify the input global data
# sourceLocation = '/home/pliu/Downloads/MipREA_DATA/batch_MipREA_global.xlsx'
# #sourceLocation = 'test.xlsx'
# #Specify the output file path
# targetLocation = "test_all.xlsx"
#
# #Specify the separator of the variable and visits
# separator = '_'
#
# #Specify the variable(column) you need to conserve
# columnNameList = ["EBV (cp/mL)", "CMV (cp/mL)", "HSV1 (cp/mL)", "HHV6 (cp/mL)", "HHV7 (Ct)", "TTV (cp/mL)"]
#
# #columnNameList = ["IC (Ct)", u"CMV (cp/µL)"]
#
# #call the main curation function
# result = curateGlobal(sourceLocation, columnNameList)
#
# #write to excel or csv
# result.to_excel(targetLocation, index=False, header=True)



####################################
####Test replace empty with specific value############
####################################
str_cell_value = "SNO_VALUE"
digit_cell_value = '.'
dataFrame = pd.read_excel("test_all.xlsx", 0)
#test = fillEmptyStringCell(dataFrame, "SEX", str_cell_value)
test1 = fillAllEmptyDigitalCell(dataFrame, digit_cell_value)
#print(test)
print(test1)
test1.to_csv('miprea_viral_reactivation.csv', sep=";", index=False, header=True, encoding='utf-8')
####################################
####Curate Miprea Global############
####################################