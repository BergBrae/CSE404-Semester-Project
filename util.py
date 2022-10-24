from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import CategoricalDtype
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def categoryLimit(col: pd.Series, count: int) -> pd.Series:
    """
    Limit the number of categories in a column to count.
    All others are 'other'.
    """
    col = col.copy()
    if col.dtype.name != 'category' or col.name == 'target':
        return col
    # get the names of the top categories
    topCategories = col.value_counts().nlargest(count-1).index 
    # limit the categories to ontly the top categories
    col = col.astype(CategoricalDtype(categories=list(topCategories) + ['other'])) 
    # replace all categories not in the top categories with 'other'
    col.fillna('other', inplace=True)

    return col

def dfCategoryLimit(df: pd.DataFrame, count: int) -> pd.DataFrame:
    """
    Limit the number of categories in a dataframe to count.
    All others are 'other'.
    """
    return df.apply(lambda col: categoryLimit(col, count))

# build a column transformer
# include a custom function to limit the number of categories in a column
def getPreprocessor():
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include='category')),
            ('num', StandardScaler(), make_column_selector(dtype_exclude='category'))
        ],
    )

def XYSplit(train, test):
    XTrain = train.drop(columns=['target'])
    YTrain = train['target'].map({'<=50K': 0, '>50K': 1})

    XTest = test.drop(columns=['target'])
    YTest = test['target'].map({'<=50K': 0, '>50K': 1})

    return XTrain, YTrain, XTest, YTest

def donwSampleLabels(X, y):
    """
    Downsample the labels to make the data more balanced.
    """
    # get the number of samples in each class
    numSamples = y.value_counts()
    # get the minimum number of samples
    minSamples = numSamples.min()
    # get the index of the samples to keep
    keepIdx = y.groupby(y).apply(lambda x: x.sample(minSamples, random_state=0, replace=False)).index.get_level_values(1)
    # get the downsampled data
    X = X.loc[keepIdx]
    y = y.loc[keepIdx]

    return X, y

def getMetrics(clf, XTest, YTest):
    """
    Get the metrics for the test data.
    """
    YPred = clf.predict(XTest)
    YPredProb = clf.predict_proba(XTest)[:, 1]

    metrics = {
        'accuracy': accuracy_score(YTest, YPred),
        'precision': precision_score(YTest, YPred),
        'recall': recall_score(YTest, YPred),
        'f1': f1_score(YTest, YPred),
        'roc_auc': roc_auc_score(YTest, YPredProb)
    }
    return metrics

def plotConfusionMatrix(yTrue, yPred, labels, title, savePath=None):
    # ex: plotConfusionMatrix(YTrain, YTrainPred, ['<=50K', '>50K'], 'Train')
    cm = confusion_matrix(yTrue, yPred, normalize='true')
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.show()
    if savePath is not None:
        fig.savefig(savePath, bbox_inches='tight')
