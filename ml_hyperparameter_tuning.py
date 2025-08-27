import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



def calcLoss(yActual, yPredicted):
    # compute confusion matrix parameters (TP, TN, FP, FN)
    correct = 0
    for i in range(len(yActual)):
        actual = yActual[i]
        predicted = yPredicted[i]
        if (predicted == actual):
            correct = correct + 1

    return 1-(correct/len(yActual))

df = pd.read_excel("data.xlsx")
df.drop(columns=["Unnamed: 0"], inplace=True)

moonX = df.loc[:,['moonx0', 'moonx1']].values
moony = df["moony"].values
circleX = df.loc[:,['circlex0', 'circlex1']].values
circley = df["circley"].values
blobX = df.loc[:,['blobx0', 'blobx1']].values
bloby = df["bloby"].values


#will only use moon, using the others is very simple as well
Xtrain, Xtest, ytrain, ytest = train_test_split(moonX, moony, test_size=0.3, random_state=42, shuffle=True)

ks = []
L = []
for k in range(20):
    if k % 2 == 1:
        ks.append(k)
        knn = KNeighborsClassifier(k)
        knn.fit(Xtrain, ytrain)
        ypred = knn.predict(Xtest)
        L.append(calcLoss(ytest,ypred))

print("The k with the least loss is", ks[L.index(min(L))], "With the loss value of ", min(L))


maxd = []
L = []
for m in range(20)[1:]:
    dt = DecisionTreeClassifier(max_depth=5)
    maxd.append(m)
    dt.fit(Xtrain, ytrain)
    ypred = dt.predict(Xtest)
    L.append(calcLoss(ytest, ypred))

print("The max depth with the least loss is", maxd[L.index(min(L))], "With the loss value of ", min(L))

maxd = np.array(range(20))[1:]
N = np.array(range(25))[5:]
L = np.zeros([len(N), len(maxd)])
i=0
for n in N:
    for m in range(len(maxd)):
        rfc = RandomForestClassifier(max_depth=maxd[m], n_estimators=n)
        rfc.fit(Xtrain, ytrain)
        ypred = rfc.predict(Xtest)
        L[i, m] = calcLoss(ytest, ypred)
    i = i + 1

Nindex = np.where(L == L.min())[0][0]
maxDIndex = np.where(L == L.min())[1][0]

print("The max depth and n estimators combintaion with the least loss is [", maxd[maxDIndex], N[Nindex], "] With the loss value of ", L.min())


C = []
L = []
for c in np.array(range(100))[1:]/100:
    s = SVC(kernel="linear", C=c)
    C.append(c)
    s.fit(Xtrain, ytrain)
    ypred = s.predict(Xtest)
    L.append(calcLoss(ytest, ypred))

print("The C with the least loss is", C[L.index(min(L))], "With the loss value of ", min(L))

G = np.array(range(10))[1:]
L = np.zeros([3, len(G)])
kernels = ['poly', 'rbf', 'sigmoid']
i=0
for k in kernels:
    for g in range(len(G)):
        s = SVC(gamma=G[g], C=1, kernel=k)
        s.fit(Xtrain, ytrain)
        ypred = s.predict(Xtest)
        L[i, g] = calcLoss(ytest, ypred)
    i = i + 1

kernelInd = np.where(L == L.min())[0][0]
GIndex = np.where(L == L.min())[1][0]

print("The Gamma and kernel combintaion with the least loss is [", G[GIndex], kernels[kernelInd], "] With the loss value of ", L.min())





