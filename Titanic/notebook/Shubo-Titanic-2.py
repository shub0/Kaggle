import pandas as pd
import numpy as np



def names(train, test):
    for dataset in [train, test]:
        dataset["NameLen"] = dataset["Name"].apply(lambda x: len(x))
        dataset["Title"] = dataset["Name"].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        del dataset["Name"]
    return train, test

def age_impute(train, test):
    for dataset in [train, test]:
        dataset["HasAge"] = dataset["Age"].apply(lambda x: 1 if pd.isnull(x) else 0)
        ages = train.groupby(["Title", "Pclass"])["Age"]
        dataset["Age"] = ages.transform(lambda x: x.fillna(x.mean()))
    return train, test

def family_size(train, test):
    for dataset in [train, test]:
        dataset["FamilySize"] = np.where(dataset["SibSp"] + dataset["Parch"] == 0, "Individual",
                                     np.where( (dataset["SibSp"] + dataset["Parch"]) <= 3, "Small", "Big") )
        del dataset["SibSp"]
        del dataset["Parch"]
    return train, test

def ticket_grouped(train, test):
    for dataset in [train, test]:
        dataset['TicketLett'] = dataset['Ticket'].apply(lambda x: str(x)[0])
        dataset['TicketLett'] = dataset['TicketLett'].apply(lambda x: str(x))
        dataset['TicketLett'] = np.where((dataset['TicketLett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), dataset['TicketLett'],
                                   np.where((dataset['TicketLett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        dataset['TicketLen'] = dataset['Ticket'].apply(lambda x: len(str(x)))
        del dataset['Ticket']
    return train, test

def cabin(train, test):
    for dataset in [train, test]:
        dataset["CabinLetter"] = dataset["Cabin"].apply(lambda x: str(x)[0])
        del dataset["Cabin"]
    return train, test

def cabin_num(train, test):
    for dataset in [train, test]:
        dataset['CabinNum1'] = dataset['Cabin'].apply(lambda x: str(x).split(" ")[-1][1:])
        dataset['CabinNum1'].replace('an', np.NaN, inplace = True)
        dataset['CabinNum1'] = dataset['CabinNum1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
        dataset['CabinNum'] = pd.qcut(train['CabinNum1'],3)

    train = pd.concat((train, pd.get_dummies(train['CabinNum'], prefix = 'CabinNum')), axis = 1)
    test = pd.concat((test, pd.get_dummies(test['CabinNum'], prefix = 'CabinNum')), axis = 1)
    del train['CabinNum']
    del test['CabinNum']
    del train['CabinNum1']
    del test['CabinNum1']
    return train, test

def embarked_impute(train, test):
    for dataset in [train, test]:
        # most frequent port
        dataset["Embarked"] = dataset["Embarked"].fillna("S")
    return train, test

# categorical columns into dummy variables.
def dummies(train, test, columns):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column + '_' + i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test

# Drop the PassengerId column
def drop(train, test, droppedCols = ['PassengerId']):
    for dataset in [train, test]:
        for col in droppedCols:
            del dataset[col]
    return train, test

def check_null(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    stats = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return stats


def run_kfold(clf, n_folds = 10):
    from sklearn.cross_validation import KFold
    from sklearn.metrics import accuracy_score
    train_size = train.shape[0]
    kf = KFold(train_size, n_folds)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


train, test = names(train, test)
train, test = age_impute(train, test)
train, test = cabin_num(train, test)
train, test = cabin(train, test)
train, test = embarked_impute(train, test)
train, test = family_size(train, test)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
train, test = ticket_grouped(train, test)
columns = ['Pclass', 'Sex', 'Embarked', 'TicketLett', 'CabinLetter', 'Title', 'FamilySize']
train, test = dummies(train, test, columns)
train, test = drop(train, test)

y = train["Survived"]
X = train.drop(["Survived"], axis = 1)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(X, y)
run_kfold(rf, 5)

'''
# SVM
from sklearn.svm import SVC, LinearSVC
svc = SVC(kernel = "rbf", C = 10)
svc.fit(X, y)
run_kfold(svc)

'''
# GBT
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             random_state=1)

gbc.fit(X, y)
run_kfold(gbc, 5)

'''
feature_imp = pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ["variable"]),
                         pd.DataFrame(rf.feature_importances_, columns = ["importance"])),
                        axis = 1).sort_values(by="importance", ascending = False)
print(feature_imp)
'''
predictions = gbc.predict(test)
predictions = pd.DataFrame(predictions, columns=["Survived"])
test = pd.read_csv("../input/test.csv")
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv("../output/gbc-result.csv", sep=",", index = False)
