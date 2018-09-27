import csv

def readStream(baseName,iter=0):
    with open("%s_%s.csv"%(baseName,str(iter))) as f:
        reader = csv.reader(f)
        if iter == 0:
            returnList = []
            for a in list(reader):
                returnList.append(a[:2])
            return returnList
        return list(reader)