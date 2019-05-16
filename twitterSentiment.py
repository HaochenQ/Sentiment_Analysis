from sklearn.metrics import classification_report as report
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import csv


def getLabel(list,filename):
    with open(filename,'r')as f:
        for line in f:
            line=line.replace("\n","").split("\t")  # split can convert String line to array
            list.append(line[1])
    return list
label_train=getLabel([],'train-labels.txt')
label_eval=getLabel([],'eval-labels.txt')

# convert (positive/negative/nutural) into machine understandable language
'''for each in range(len(label)):
    if label[each]=='positive':
        label[each] = 2
    elif label[each] == 'negative':
        label[each] = 0
    else: label[each] = 1'''

# print(label)
# csv_file=open('train.csv',encoding='utf-8')
# train = csv.reader(csv_file)
# print(train)

df1=pd.read_csv('train.csv',sep=',')
df1=df1.drop('id',axis=1)
df2=pd.read_csv('eval.csv',sep=',')
df2=df2.drop('id',axis=1)
df3=pd.read_csv('test.csv',sep=',')
top5=df1.head()
tail5=df1.tail()
col=df1.columns
#print(top5)
#print(tail5)
#print(col)

# leveraging of SVM
model_svm=svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
model_svm.fit(df1,label_train)
result_svm=model_svm.predict(df2)




# print(list(result_eval))


# leveraging of Decision Tree
model_Dtree=tree.DecisionTreeClassifier()
model_Dtree.fit(df1,label_train)
result_tree=model_Dtree.predict(df2)


# leveraging of Naive Bayes
model_NB = GaussianNB()
model_NB.fit(df1,label_train)
result_NB = model_NB.predict(df2)


# random forest
model_Rforest = RandomForestClassifier(n_estimators=15) # num of trees in forest
model_Rforest.fit(df1,label_train)
result_Rf = model_Rforest.predict(df2)

# NN
model_nn=KNeighborsClassifier(n_neighbors=5)
model_nn.fit(df1,label_train)
result_nn = model_nn.predict(df2)



# print all result
report1 = report(label_eval,result_svm)
report2 = report(label_eval,result_tree)
report3 = report(label_eval,result_NB)
report4 = report(label_eval,result_Rf)
report5 = report(label_eval,result_nn)
print(report1)
print('\n',report2)
print('\n',report3)
print('\n',report4)
print('\n',report5)
