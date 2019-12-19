import logging
import numpy as np
import pandas as pd
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import sklearn
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm, skew
from scipy.special import boxcox1p

# Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

logger = logging.getLogger(__name__)


def preprocessing(**kwargs):

    logging.info('variables successfully fetched from previous task')
    logging.info('Start preprocessing data')

    # read csv
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    train_ID = train['Id']
    test_ID = test['Id']

    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)

    # Deleting outliers
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

    train["SalePrice"] = np.log1p(train["SalePrice"])
    # concat train and test data for preprocessing
    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    logging.info("all_data size is : {}".format(all_data.shape))

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'MiscFeature', 'Alley',
                'Fence', 'FireplaceQu', 'MasVnrType'):
        all_data[col] = all_data[col].fillna('None')

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)

    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
        all_data[col] = all_data[col].fillna(0)

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MSSubClass'):
        all_data[col] = all_data[col].fillna('None')

    for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'):
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    for col in ('MSSubClass', 'OverallCond', 'YrSold', 'MoSold'):
        all_data[col] = all_data[col].astype(str)

    all_data = all_data.drop(['Utilities'], axis=1)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")

    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))

    # shape
    logging.info('Shape all_data: {}'.format(all_data.shape))

    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    logging.info("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew': skewed_feats})

    skewness = skewness[abs(skewness) > 0.75]
    logging.info("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        all_data[feat] = boxcox1p(all_data[feat], lam)

    all_data = pd.get_dummies(all_data)
    logging.info(all_data.shape)

    train = all_data[:ntrain]
    label = y_train
    test = all_data[ntrain:]

    return[train, label, test, test_ID]
