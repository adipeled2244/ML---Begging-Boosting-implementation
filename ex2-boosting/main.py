import pandas as pd
import numpy as np

from sklearn import tree
import statistics

def adaBoost():
    trainingData = pd.read_csv("titanikData.csv")
    testData = pd.read_csv('titanikTest.csv', names=["pclass", "age", "gender", "survived"])

    # make training data with numbers
    changeCategoryValueToNumberForTwoValues(trainingData,'age','child','adult')
    changeCategoryValueToNumberForTwoValues(trainingData, 'gender', 'female', 'male')
    changeCategoryValueToNumberForTwoValues(trainingData, 'survived', 'yes', 'no')
    changeCategoryValueToNumberForFourValues(trainingData, 'pclass','1st', '2nd', '3rd','crew')

    # intiialize whigh to 1/n
    trainingData["weights"]=1/len(trainingData)

    dataWithoutTarget = trainingData.drop(['survived'], axis=1)
    dataWithoutTarget=dataWithoutTarget.drop(['weights'], axis=1)
    target = trainingData['survived']

    modelsTreeTosave=[]
    alphas=[]
    numIterations = 3

    for i in range(0, numIterations):
        treeModel = createTreeModel(dataWithoutTarget, target,trainingData)
        modelsTreeTosave.append(treeModel)

        trainingData['predictions']=treeModel.predict(dataWithoutTarget)
        trainingData['isSame']=trainingData['predictions']==trainingData['survived']

        # calculate errorRate
        correct = 0
        incorrect = 0
        for idx, example in enumerate(trainingData['isSame']):
            if (example == True):
                # print(trainingData.iloc[idx])
                correct = correct + trainingData.iloc[idx].weights
            else:
                incorrect = incorrect + trainingData.iloc[idx].weights
        # print('correct', correct, ' incorrect', incorrect, )
        errorRate=incorrect

        beta= errorRate/(1-errorRate)

        # change weight for true examples by *beta
        for idx, simple in enumerate(trainingData['isSame']):
            if (simple):
                trainingData.at[idx, 'weights'] = trainingData.iloc[idx].weights * beta

        # rescale by dividing
        sumWeights = trainingData['weights'].sum()
        weightToAddtoEveryone = (1 - sumWeights) / len(trainingData)

        trainingData['weights'] = trainingData['weights'] + weightToAddtoEveryone

        alpah= 0.5 * np.log(1 / beta)
        alphas.append(alpah)

    # make test data with numbers
    changeCategoryValueToNumberForTwoValues(testData,'age','child','adult')
    changeCategoryValueToNumberForTwoValues(testData, 'gender', 'female', 'male')
    changeCategoryValueToNumberForTwoValues(testData, 'survived', 'yes', 'no')
    changeCategoryValueToNumberForFourValues(testData, 'pclass','1st', '2nd', '3rd','crew')

    testWithoutTarget = testData.drop(['survived'], axis=1)

    # prediction on test
    predictionsFromTestFinal=[]
    for idx1,modelTree in enumerate(modelsTreeTosave):
        prediction=modelTree.predict(testWithoutTarget)
        for idx2, i in enumerate(prediction):
            if (i == 0):
                prediction[idx2] = -1
        prediction = alphas[idx1] * prediction
        predictionsFromTestFinal.append(prediction)

    print(predictionsFromTestFinal)

    resultsSigns=[]
    for numElemenet in range(len(predictionsFromTestFinal[0])):
        sum = 0
        for predict in predictionsFromTestFinal:
           sum=sum+predict[numElemenet]
           # print(predict[numElemenet])
        # print("sum",numElemenet,sum)
        resultsSigns.append(np.sign(sum))


    testData['predResult']=resultsSigns

    print(testData)
    testData = changeNumValueToCatForTwoValues(testData, 'age', 'child', 'adult')
    testData = changeNumValueToCatForTwoValues(testData, 'gender', 'female', 'male')
    testData = changeNumValueToCatForTwoValues(testData, 'survived', 'yes', 'no')
    testData = changeNumValueToCatForFourValues(testData, 'pclass', '1st', '2nd', '3rd', 'crew')
    testData =changeNumValueToCatForTwoValuesForPred(testData, 'predResult', 'yes', 'no')
    print(testData)

    testData['isSame'] = testData["predResult"] == testData["survived"]

    isSame = testData['isSame'].value_counts()
    success = isSame[True] / (isSame[True] + isSame[False])
    success = success * 100
    print("success: ")
    print(success, "%")
    print("Test Data results: ")
    print(testData)

def changeCategoryValueToNumberForTwoValues(data,col,value1,value2):
    mapping={value1: 1,value2:0}
    data[col]=data[col].map(mapping);
    return data;

def changeCategoryValueToNumberForFourValues(data,col,value1,value2,value3,value4):
    mapping={value1: 1,value2:2,value3:3,value4:4}
    data[col]=data[col].map(mapping);
    return data;


def createTreeModel(data, targets,training):
    model_tree = tree.DecisionTreeClassifier()
    model_tree.criterion="entropy"
    model_tree.max_depth=2
    final_model=model_tree.fit(data,targets, sample_weight=np.array(training['weights']))
    return final_model

def changeNumValueToCatForTwoValues(test,col,value1,value2):
    mapping={1: value1 ,0:value2}
    test[col]=test[col].map(mapping);
    return test;

def changeNumValueToCatForTwoValuesForPred(test,col,value1,value2):
    mapping={1: value1 ,-1:value2}
    test[col]=test[col].map(mapping);
    return test;

def changeNumValueToCatForFourValues(test,col,value1,value2,value3,value4):
    mapping={1: value1,2:value2,3:value3,4:value4}
    test[col]=test[col].map(mapping);
    return test;


if __name__ == "__main__":
    adaBoost()


