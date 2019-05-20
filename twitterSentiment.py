from sklearn.metrics import classification_report as report
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import csv
import time
initTime=time.time()
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



# feature selection
#df1=SelectPercentile(chi2,percentile=99).fit_transform(df1,label_train)
#df2=SelectPercentile(chi2,percentile=99).fit_transform(df2,label_eval)

'''# leveraging of SVM
model_svm=svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=1, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
model_svm.fit(df1,label_train)
result_svm=model_svm.predict(df2)

report1 = report(label_eval,result_svm,digits=5)
print(report1)


model_svm=svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=2, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
model_svm.fit(df1,label_train)
result_svm=model_svm.predict(df2)

report11 = report(label_eval,result_svm,digits=5)
print(report11)


model_svm=svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='scale', kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
model_svm.fit(df1,label_train)
result_svm=model_svm.predict(df2)

report12 = report(label_eval,result_svm,digits=5)
print(report12)


# leveraging of Decision Tree
model_Dtree=tree.DecisionTreeClassifier()
model_Dtree.fit(df1,label_train)
result_tree=model_Dtree.predict(df2)

report2 = report(label_eval,result_tree,digits=5)
print('\n',report2)




# leveraging of Naive Bayes
model_NB = MultinomialNB()
model_NB.fit(df1,label_train)
result_NB = model_NB.predict(df2)

report3 = report(label_eval,result_NB,digits=5)
print('\n',report3)
'''

# random forest
model_Rforest = RandomForestClassifier(n_estimators=10,random_state=10) # num of trees in forest
model_Rforest.fit(df1,label_train)
result_Rf = model_Rforest.predict(df2)

report4 = report(label_eval,result_Rf,digits=5)
print('\n',report4)


model_Rforest = RandomForestClassifier(n_estimators=10,random_state=100) # num of trees in forest
model_Rforest.fit(df1,label_train)
result_Rf = model_Rforest.predict(df2)

report41 = report(label_eval,result_Rf,digits=5)
print('\n',report41)


model_Rforest = RandomForestClassifier(n_estimators=10,random_state=200) # num of trees in forest
model_Rforest.fit(df1,label_train)
result_Rf = model_Rforest.predict(df2)

report42 = report(label_eval,result_Rf,digits=5)
print('\n',report42)
'''
# NN
model_nn=KNeighborsClassifier(n_neighbors=20)
model_nn.fit(df1,label_train)
result_nn = model_nn.predict(df2)

report5 = report(label_eval,result_nn,digits=5)
print('\n',report5)


model_nn=KNeighborsClassifier(n_neighbors=25)
model_nn.fit(df1,label_train)
result_nn = model_nn.predict(df2)

report51 = report(label_eval,result_nn,digits=5)
print('\n',report51)


model_nn=KNeighborsClassifier(n_neighbors=50)
model_nn.fit(df1,label_train)
result_nn = model_nn.predict(df2)

report52 = report(label_eval,result_nn,digits=5)
print('\n',report52)
'''