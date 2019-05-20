import pandas as pd

def addcol(f1,f2,newfile):
    list=[]
    with open(f2,'r')as f:
        for line in f:
            line=line.replace("\n","").split("\t")  # split can convert String line to array
            list.append(line[1])
    df = pd.read_csv(f1)
    df['label']= list
    df.to_csv(newfile,index=0,header=1)

#addcol('train.csv','train-labels.txt','new-train.csv')
addcol('eval.csv','eval-labels.txt','new-eval.csv')