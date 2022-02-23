import pandas as pd
import numpy as np

# from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import statistics

def bagging():
    trainingData = pd.read_csv("titanikData.csv")
    testData = pd.read_csv('titanikTest.csv', names=["pclass", "age", "gender", "survived"])

    # make training data with numbers
    changeCategoryValueToNumberForTwoValues(trainingData,'age','child','adult')
    changeCategoryValueToNumberForTwoValues(trainingData, 'gender', 'female', 'male')
    changeCategoryValueToNumberForTwoValues(trainingData, 'survived', 'yes', 'no')
    changeCategoryValueToNumberForFourValues(trainingData, 'pclass','1st', '2nd', '3rd','crew')

    # create bootsrps
    unique_records = trainingData.drop_duplicates()
    bootsraps=createBootsraps(trainingData,unique_records,100)

    # create 100 tree models from 100 bootsraps
    treesModels=[]
    treesModels=createTreeModels(100,bootsraps,"survived");

    # make test data with numbers
    changeCategoryValueToNumberForTwoValues(testData,'age','child','adult')
    changeCategoryValueToNumberForTwoValues(testData, 'gender', 'female', 'male')
    changeCategoryValueToNumberForTwoValues(testData, 'survived', 'yes', 'no')
    changeCategoryValueToNumberForFourValues(testData, 'pclass','1st', '2nd', '3rd','crew')

    #get targets from all models about all exaples
    predictions=[]
    predictions = makePredictsOnTest(treesModels,testData,"survived")

    #mode on the results of all trees
    resultsCurr = []
    resultsOfTargets=[]

    for numElemenet in range(len(predictions[0])):
        for predict in predictions:
            resultsCurr.append(predict[numElemenet])
        resultsOfTargets.append(statistics.mode(resultsCurr))
        resultsCurr=[]

    # add the targets from prediction to test table , change test from numbers to categort cols
    testData["resultsPred"]=resultsOfTargets
    testData=changeNumValueToCatForTwoValues(testData,'age','child', 'adult')
    testData=changeNumValueToCatForTwoValues(testData, 'gender', 'female', 'male')
    testData=changeNumValueToCatForTwoValues(testData, 'survived', 'yes', 'no')
    testData=changeNumValueToCatForTwoValues(testData, 'resultsPred', 'yes', 'no')

    testData=changeNumValueToCatForFourValues(testData, 'pclass', '1st', '2nd', '3rd', 'crew')


    testData['isSame'] = testData["resultsPred"] == testData["survived"]

    isSame= testData['isSame'].value_counts()
    success=isSame[True]/(isSame[True]+isSame[False])
    success=success*100
    print("success: ")
    print(success,"%")
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

# return models of decisions tree
def createBootsraps(trainingData, uniqueData, numBootsraps):
    bootsraps = []
    for i in range(numBootsraps):
        bootsrapUnique = uniqueData.sample(frac=0.63)
        bootstrapWithDuplicates = bootsrapUnique.sample(n=len(trainingData) - len(bootsrapUnique), replace=True)
        unionBootstrap = [bootsrapUnique, bootstrapWithDuplicates]
        bootsraps.append(pd.concat(unionBootstrap))
    return bootsraps;

def createTreeModel(data, targets):
    model_tree = tree.DecisionTreeClassifier()
    model_tree.criterion="entropy"
    model_tree.max_depth=1
    final_model=model_tree.fit(data,targets)
    return final_model

def createTreeModels(numTrees,bootstraps,target):
   allTreesModels=[]
   for i in range(numTrees):
        data = bootstraps[i].drop([target], axis=1)
        targets = bootstraps[i][target]
        allTreesModels.append(createTreeModel(data, targets))
   return allTreesModels

def makePredictsOnTest(treesModels,testData,col):
    predictions = []
    for i, treeModel in enumerate(treesModels):
        testWithoutTargets = testData.drop([col], axis=1)
        predictions.append(treeModel.predict(testWithoutTargets))
    return predictions


def changeNumValueToCatForTwoValues(test,col,value1,value2):
    mapping={1: value1 ,0:value2}
    test[col]=test[col].map(mapping);
    return test;

def changeNumValueToCatForFourValues(test,col,value1,value2,value3,value4):
    mapping={1: value1,2:value2,3:value3,4:value4}
    test[col]=test[col].map(mapping);
    return test;

if __name__ == "__main__":
    bagging()


