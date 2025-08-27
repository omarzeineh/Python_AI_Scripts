import numpy as np
from statistics import mode

def knn(xtrain, ytrain, xtest, k):
    # normalize the dataset

    maxValue = np.max(np.concatenate((xtrain, [xtest])),axis=0)
    minValue = np.min(np.concatenate((xtrain, [xtest])),axis=0)

    xtrain = (xtrain - minValue) / (maxValue - minValue)
    xtest = (xtest - minValue) / (maxValue - minValue)

    # compute distances between xtrain and xtest
    distance = np.sqrt(np.sum((xtrain - xtest) * (xtrain - xtest), axis=1))

    PredectedClass = -1
    votes = 0
    votelist=[]
    
    # in the loop, we are finding the indices of minimum k distances, and then we are doing majority voting
    for i in range(k):
        index = np.argmin(distance)
        distance[index] = 100000
        votelist.append(ytrain[index])
    
    PredectedClass=mode(votelist)  #mode function is used to find the most common element in the list
    
    return int(PredectedClass)

def computePerformanceMetrics(yActual, yPredicted, pos_class):
    # compute confusion matrix parameters (TP, TN, FP, FN)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(yActual)):
        actual = yActual[i]
        predicted = yPredicted[i]
       
        if (predicted == 1) and (actual == 1):
            TP = TP + 1
        elif (predicted == 0) and (actual == 1):
            FN = FN + 1
        elif (predicted == 1) and (actual == 0):
            FP = FP + 1
        elif (predicted == 0) and (actual == 0):
            TN = TN + 1


    # compute accuracy, recall, precision, F1
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    FNR = FN/(TP+FN)
    FPR = FP/(FP+TN)
    SPC = TN/(FP+TN)
    F1 = (2 * precision * recall) / (precision + recall)


    confusion_matrix = np.array([[TP, FN],[FP, TN]])

    print("Confusion Matrix: ")
    print(confusion_matrix)
    print("Accuracy", accuracy)
    print("Precision", precision)
    print("Recall", recall)
    print("F1", F1)
    print("FNR", FNR)
    print("FPR", FPR)
    print("SPC", SPC)



data = np.array([[7.3,2.9,0],
                [5.9,3.0,0],
                [6.7,3.3,0],
                [4.4,2.9,1],
                [5.3,3.7,1],
                [7.7,2.8,0],
                [5.6,2.5,0],
                [7.7,3.0,0],
                [5.7,2.8,0],
                [5.6,2.8,0],
                [4.7,3.2,1],
                [5.1,3.7,1],
                [6.3,2.5,0],
                [4.8,3.4,1],
                [4.9,3.1,1],
                [5.0,3.4,1],
                [5.5,2.4,0],
                [6.3,3.3,0],
                [6.4,3.1,0],
                [5.1,3.8,1],
                [5.4,3.7,1],
                [6.3,2.9,0],
                [6.7,3.0,0],
                [5.2,3.4,1],
                [5.1,3.8,1],
                [5.0,3.5,1],
                [6.5,2.8,0],
                [5.7,3.8,1],
                [6.7,2.4,1],
                [4.4,3.2,0],
                ])

xtrain = data[0:20, 0:2]
ytrain = data[0:20, 2]
xtest = data[20:30, 0:2]
ytest = data[20:30, 2]

actualOutput = []
predictedOutput = []
for i in range(0,10):
    actualOutput.append(int(ytest[i]))
    predictedOutput.append(knn(xtrain, ytrain, xtest[i],3))

print("Actual Test Sample Output: ", actualOutput)
print("Predicted Test Sample Output: ", predictedOutput)

computePerformanceMetrics(np.array(actualOutput), np.array(predictedOutput), 1)
