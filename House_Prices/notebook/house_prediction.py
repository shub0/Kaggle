# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

def fill_na(train, test, columns):
    for column in columns:
        if(train[column].dtype == np.float64 or train[column].dtype == np.int64):
            print("CANNOT fill column: %s" % (column))
        else:
            print("Fill column %s" % (column))
            train[column].fillna(value = "NA", inplace=True)
            test[column].fillna(value = "NA", inplace=True)

def fill_value(train, test, columns):
    for column in columns:
        if(train[column].dtype == np.object):
            mode_value = pd.concat( [train[column].dropna(), test[column].dropna()], axis = 0).mode()
            print("Fill na with mode (%s) for %s" % (str(mode_value.values[0]), column))
            train[column].fillna(value = mode_value.values[0], inplace=True)
            test[column].fillna(value = mode_value.values[0], inplace=True)
        else:
            median_value = train[column].dropna().median()
            print("Fill na with mean (%f) for %s" % (float(median_value.values[0]), column))
            train[column].fillna(value = float(median_value.values[0]), inplace=True)
            test[column].fillna(value = float(median_value.values[0]), inplace=True)

# Set frontage by Neighborhood
def set_frontage(train, test):
    lot_frontage_by_neighborhood = train["LotFrontage"].groupby(train["Neighborhood"])
    for key, group in lot_frontage_by_neighborhood:
        idx = (train["Neighborhood"] == key) & (train["LotFrontage"].isnull())
        train.loc[idx, "LotFrontage"] = group.median()
        idx = (test["Neighborhood"] == key) & (test["LotFrontage"].isnull())
        test.loc[idx, "LotFrontage"] = group.median()

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

test.loc[ pd.isnull(test["GarageCars"]), "GarageType"] = "NA"
test["GarageCars"].fillna(value = 0, inplace = True)
test["GarageArea"].fillna(value = 0, inplace = True)
train["Target"] = np.log(train["SalePrice"])

for col in ["GarageType", "GarageQual", "GarageCond"]:
    train[col].fillna(value = "NA", inplace=True)
    test[col].fillna(value = "NA", inplace=True)

train["GarageYrBlt"].fillna(train["YearBuilt"], inplace=True)
test["GarageYrBlt"].fillna(test["YearBuilt"], inplace=True)
train["GarageFinish"].fillna(value="NA", inplace=True)
test["GarageFinish"].fillna(value="NA", inplace=True)

bsmt_cols = filter(lambda col: col.find("Bsmt") >= 0, train.columns)
for col in bsmt_cols:
    if train[col].dtype == np.object:
        train[col].fillna(value = "NA", inplace = True)
        test[col].fillna(value = "NA", inplace = True)
    else:
        train[col].fillna(value = 0, inplace = True)
        test[col].fillna(value = 0, inplace = True)

train["FireplaceQu"].fillna(value = "NA", inplace = True)
test["FireplaceQu"].fillna(value = "NA", inplace = True)

train.loc[ (train["MasVnrArea"] == 0), "MasVnrType"] = "None"
train["MasVnrArea"].fillna(value = 0, inplace = True)
train["MasVnrType"].fillna(value = "None", inplace = True)

test.loc[ (pd.isnull(test["MasVnrType"])), "MasVnrArea" ] = 0
test["MasVnrArea"].fillna(value = 0, inplace = True)
test["MasVnrType"].fillna(value = "None", inplace = True)

test["Functional"].fillna(value = "Typ", inplace=True)

set_frontage(train, test)
high_na_columns = [u'PoolQC', u'MiscFeature', u'Alley', u'Fence', u'FireplaceQu']
fill_na(train, test, high_na_columns)
null_features = [u"Electrical", u'Exterior1st', u'Exterior2nd', u'Utilities',
               u'MSZoning', u'Functional', u'SaleType']
fill_value(train, test, null_features)

train["PoolQC"].fillna(value = 0, inplace=True)
test["PoolQC"].fillna(value = 0, inplace=True)

year_map = pd.concat(pd.Series("YearBin" + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))

# Set discrete string value to numerical value
def munge_quality(train, test):
    qual_dict = {None: 0, "NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    bsmt_exposure_dict = {None: 0, "NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
    bsmt_fin_dict = {None: 0, "NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    function_dict = {None: 0, "NA": 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}
    fence_dict = {None: 0, "NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}
    finish_dict = {None: 0,"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3}

    for dataset in [train, test]:
        dataset["ExterQual"] = dataset["ExterQual"].map(qual_dict).astype(int)
        dataset["ExterCond"] = dataset["ExterCond"].map(qual_dict).astype(int)
        dataset["BsmtQual"] = dataset["BsmtQual"].map(qual_dict).astype(int)
        dataset["BsmtCond"] = dataset["BsmtCond"].map(qual_dict).astype(int)
        dataset["HeatingQC"] = dataset["HeatingQC"].map(qual_dict).astype(int)
        dataset["PoolQC"] = dataset["PoolQC"].map(qual_dict).astype(int)
        dataset["KitchenQual"] = dataset["KitchenQual"].map(qual_dict).astype(int)
        dataset["FireplaceQu"] = dataset["FireplaceQu"].map(qual_dict).astype(int)
        dataset["GarageQual"] = dataset["GarageQual"].map(qual_dict).astype(int)
        dataset["GarageCond"] = dataset["GarageCond"].map(qual_dict).astype(int)

        dataset["BsmtExposure"] = dataset["BsmtExposure"].map(bsmt_exposure_dict).astype(int)

        dataset["BsmtFinType1"] = dataset["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
        dataset["BsmtFinType2"] = dataset["BsmtFinType2"].map(bsmt_fin_dict).astype(int)

        dataset["Functional"] = dataset["Functional"].map(function_dict).astype(int)
        dataset["GarageFinish"] = dataset["GarageFinish"].map(finish_dict).astype(int)
        dataset["Fence"] = dataset["Fence"].map(fence_dict).astype(int)

# Set Year and Month Band
# TODO figure out best band (by DT? evenly? )
month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "June",
            7: "July", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
def set_year_band(train, test):
    for dataset in [train, test]:
        dataset["YearBuiltBin"] = dataset["YearBuilt"].map(year_map).astype(str)
        dataset["Age"] = 2010 - dataset["YearBuilt"]
        dataset["GarageYrBltBin"] = dataset["GarageYrBlt"].map(year_map).astype(str)
        dataset.loc[((dataset['MoSold'] > 2) & (dataset['MoSold'] <= 8)), 'SoldSeason'] = "HighSeason"
        dataset.loc[((dataset['MoSold'] > 8) | (dataset['MoSold'] <= 2)), 'SoldSeason'] = "LowSeason"
        dataset["MoSold"] = dataset["MoSold"].map(month_map).astype(str)
        dataset["YearRemodAddBin"] = dataset["YearRemodAdd"].map(year_map).astype(str)
        dataset["YrSoldBin"] = dataset["YrSold"].map(year_map).astype(str)
        dataset["TimeSinceSold"] = 2010 - dataset["YrSold"]

        del dataset["YrSold"]
        del dataset["YearBuilt"]
        del dataset["GarageYrBlt"]
        del dataset["YearRemodAdd"]

# Get dummies for categorical features
def set_dummies(train, test, columns = []):
    if not columns:
        category_features = filter(lambda x: train[x].dtype == object, train.columns)
    else:
        category_features = columns
    for column in category_features:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column + '_' + i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test

def set_areas(train, test):
    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]
    for dataset in [train, test]:
        area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                     'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                     'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]
        dataset["TotalArea"] = dataset[area_cols].sum(axis=1)

        dataset["TotalArea1st2nd"] = dataset["1stFlrSF"] + dataset["2ndFlrSF"]

neighborhood_map = {
    "MeadowV" : 0,  #  88000
    "IDOTRR" : 1,   # 103000
    "BrDale" : 1,   # 106000
    "OldTown" : 1,  # 119000
    "Edwards" : 1,  # 119500
    "BrkSide" : 1,  # 124300
    "Sawyer" : 1,   # 135000
    "Blueste" : 1,  # 137500
    "SWISU" : 2,    # 139500
    "NAmes" : 2,    # 140000
    "NPkVill" : 2,  # 146000
    "Mitchel" : 2,  # 153500
    "SawyerW" : 2,  # 179900
    "Gilbert" : 2,  # 181000
    "NWAmes" : 2,   # 182900
    "Blmngtn" : 2,  # 191000
    "CollgCr" : 2,  # 197200
    "ClearCr" : 3,  # 200250
    "Crawfor" : 3,  # 200624
    "Veenker" : 3,  # 218000
    "Somerst" : 3,  # 225500
    "Timber" : 3,   # 228475
    "StoneBr" : 4,  # 278000
    "NoRidge" : 4,  # 290000
    "NridgHt" : 4,  # 315000
}

for dataset in [train, test]:
    dataset["NeighborhoodBin"] = dataset["Neighborhood"].map(neighborhood_map).astype(int)

munge_quality(train, test)
set_year_band(train, test)

area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea'
            ]
for dataset in [train, test]:
    dataset["TotalSF"] = dataset[area_cols].sum(axis=1)

train_dummies_2, test_dummies_2 = set_dummies(train, test, [])
y = train_dummies_2["Target"]
X = train_dummies_2.drop( ["Target", "SalePrice", "Id"], axis = 1 )
test_dummies_2.drop([ "Id" ], axis = 1, inplace=True)

from sklearn.model_selection import train_test_split
num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_test, random_state=23)

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  mean_squared_error

parameters = {'alpha': np.logspace(-5, -1, 5) }
lasso = Lasso()
grid_search = GridSearchCV(lasso, parameters)
grid_search = grid_search.fit(X_train, y_train)
lasso = grid_search.best_estimator_

lasso.fit(X, y)
y_pred_lasso = lasso.predict(X)
y_lasso = lasso.predict(test_dummies_2)
print("Lasso %.4f" % np.sqrt(mean_squared_error(y, y_pred_lasso)))

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
# I found this best alpha through cross-validation.
parameters = {'alpha': np.logspace(-5, 5, 11) }
ridge = Ridge()

grid_search = GridSearchCV(ridge, parameters)
grid_search = grid_search.fit(X_train, y_train)
ridge = grid_search.best_estimator_

ridge.fit(X, y)
y_pred_ridge = ridge.predict(X)
y_ridge = ridge.predict(test_dummies_2)
print("Ridge %.4f" % np.sqrt(mean_squared_error(y, y_pred_ridge)))

import xgboost as xgb

xgboost = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=7200,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
xgboost.fit(X, y)
y_pred_xgboost = xgboost.predict(X)
print("XGBoost Regressor %.4f" % np.sqrt(mean_squared_error(y, y_pred_xgboost)))
y_xgboost = xgboost.predict(test_dummies_2)

y_avg = (0.5 * y_xgboost + 0 * y_ridge + 0.5 * y_lasso)
predictions = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp(y_avg)})
predictions.to_csv("../output/avg.csv", sep=",", index = False)
