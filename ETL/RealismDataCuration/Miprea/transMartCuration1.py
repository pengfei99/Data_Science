import pandas as pd
from utils import renameColumn


separator = '_'
digit_cell_value = "DNO_VALUE"

def breakToDF(dataFrame):
    sindex = 0
    dfList = []
    while(sindex < len(dataFrame.index)):
        eindex = sindex + 2
        if(eindex >= len(dataFrame.index)):
            eindex = len(dataFrame.index) - 1
#        print("End index: "+str(eindex))
        patientID = dataFrame.iloc[sindex]['Patient_ID']
        if (patientID == dataFrame.iloc[eindex]['Patient_ID']):
            tmpDF = dataFrame[sindex:eindex+1]
            dfList.append(tmpDF)
            sindex = eindex + 1
        elif (patientID == dataFrame.iloc[eindex-1]['Patient_ID']):
            eindex = eindex - 1
            tmpDF = dataFrame[sindex:eindex + 1]
            dfList.append(tmpDF)
            sindex = eindex + 1
        else:
            eindex = eindex - 2
            tmpDF = dataFrame[sindex:eindex + 1]
            dfList.append(tmpDF)
            sindex = eindex + 1
    return dfList


def buildColValues(colValue, rowSize):
    resultList = [colValue]
    while rowSize > 1:
        resultList.append(digit_cell_value)
        rowSize=rowSize-1
    return resultList


# This method take a dataframe,
def mergeLineToColumn(dataFrame,columnName):
    #determin row size of data frame
    rowSize = len(dataFrame.index)
    # get the columnValue list of the select column
    columnValues = dataFrame[columnName].tolist()
    # get the column number of the select column
    columnNumber = dataFrame.columns.get_loc(columnName)
    # insert the two required column
    if rowSize == 1:
        #column 1 with empty value
        tmpColName1 = columnName+separator+str(2)
        tmpColValues1 = buildColValues(digit_cell_value, rowSize)
        dataFrame.insert(columnNumber + 1, tmpColName1, tmpColValues1)
        #column 2 with empty value
        tmpColName2 = columnName + separator + str(3)
        tmpColValues2 = buildColValues(digit_cell_value, rowSize)
        dataFrame.insert(columnNumber + 2, tmpColName2, tmpColValues2)
        # column 0 is the origin one, we just rename it
        dataFrame = renameColumn(dataFrame, columnName, columnName + separator + "1")
    elif rowSize == 2:
        # column 1 with real value
        tmpColName1 = columnName + separator + str(2)
        tmpColValues1 = buildColValues(columnValues[1], rowSize)
        dataFrame.insert(columnNumber + 1, tmpColName1, tmpColValues1)
        # column 2 with empty value
        tmpColName2 = columnName + separator + str(3)
        tmpColValues2 = buildColValues(digit_cell_value, rowSize)
        dataFrame.insert(columnNumber + 2, tmpColName2, tmpColValues2)
        # column 0 is the origin one, we just rename it
        dataFrame = renameColumn(dataFrame, columnName, columnName + separator + "1")
    else:
        for i in range(1, rowSize):
            tmpColName = columnName+separator+str(i+1)
            tmpColValues = buildColValues(columnValues[i],rowSize)
            dataFrame.insert(columnNumber+i, tmpColName, tmpColValues)
            dataFrame = renameColumn(dataFrame, columnName, columnName+separator+"1")
    return dataFrame


def mergeMultiColumn(dataFrame, columnNameList):
    for columnName in columnNameList:
        dataFrame = mergeLineToColumn(dataFrame, columnName)
    return dataFrame


def clean(dataFrame):
    return dataFrame[0:1]


def concatDataFrames(dfList):
    return pd.concat(dfList)


def curateGlobal(sourceLocation, columnNameList):
    dataFrame = pd.read_excel(sourceLocation, 0)
    resultDFList = []
    dfList = breakToDF(dataFrame)
    for df in dfList:
        tmpDF = clean(mergeMultiColumn(df, columnNameList))
        resultDFList.append(tmpDF)
    return concatDataFrames(resultDFList)



