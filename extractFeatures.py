from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report as report



def getLabel(list,filename):
    with open(filename,'r')as f:
        for line in f:
            line=line.replace("\n","").split("\t")  # split can convert String line to array
            list.append(line[1])
    return list
label_train=getLabel([],'train-labels.txt')
label_eval=getLabel([],'eval-labels.txt')

x_vectorizer=HashingVectorizer()

corpus1=[]
corpus2=[]
with open ('train-tweets.txt',encoding='utf-8') as train:
    for line in train:
        line = line.replace("\n", "").split("\t")
        corpus1.append(line[1])
with open ('eval-tweets.txt',encoding='utf-8') as train:
    for line in train:
        line = line.replace("\n", "").split("\t")
        corpus2.append(line[1])
# print(corpus)

X=x_vectorizer.fit_transform(corpus1)
#X.toarray()
y_vectorizer=HashingVectorizer()
Y=y_vectorizer.fit_transform(corpus2)
#print(x_vectorizer.get_feature_names())


model_NB = MultinomialNB()
model_NB.fit(X,label_train)
result_NB = model_NB.predict(Y)

report3 = report(label_eval,result_NB,digits=5)
print('\n',report3)