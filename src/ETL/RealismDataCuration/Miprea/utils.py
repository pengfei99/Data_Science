import numpy as np


def renameColumn(dataFrame, oldName, newName):
    dataFrame = dataFrame.rename(columns={oldName: newName})
    return dataFrame

def fillEmptyStringCell(dataFrame, columnName, cellValue):
    dataFrame.loc[dataFrame[columnName].isnull(), columnName] = cellValue
    return dataFrame

def fillAllEmptyDigitalCell(dataFrame, cellValue):
    return dataFrame.replace(np.nan, cellValue, regex=True)