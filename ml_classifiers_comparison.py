
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def calcAcc(yActual, yPredicted):
    # compute confusion matrix parameters (TP, TN, FP, FN)
    correct = 0
    for i in range(len(yActual)):
        actual = yActual[i]
        predicted = yPredicted[i]
        if (predicted == actual):
            correct = correct + 1

    return correct/len(yActual)



classifiers = {
    "KNN (k=5)":        KNeighborsClassifier(5),
    "Decision Trees":   DecisionTreeClassifier(max_depth=5),
    "Random Forests":   RandomForestClassifier(max_depth=5, n_estimators=10),
    "Naive Bayes":      GaussianNB(),
    "Linear SVM":       SVC(kernel="linear", C=0.025),
    "Kernel SVM":       SVC(gamma=2, C=1)
    }
Accuracy = {
    "KNN (k=5)": None,
    "Decision Trees": None,
    "Random Forests": None,
    "Naive Bayes": None,
    "Linear SVM": None,
    "Kernel SVM": None
}


iris = load_iris()
X = iris.data[:, 0:4]
y = iris.target
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
print(ytest)

for classifiername, classifier in classifiers.items():
    clf = classifier
    clf.fit(Xtrain,ytrain)
    ypred = clf.predict(Xtest)
    acc = calcAcc(ytest, ypred)
    Accuracy[classifiername] = acc

print(Accuracy)




