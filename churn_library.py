"""
Module of functions for exploring data and fitting models for churn predicitons
"""
# import libraries
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import os
os.environ["QT_QPA_PLATFORM"]="offscreen"

CATEGORICAL_COLS = [
    "Gender", "Education_Level", "Marital_Status", "Income_Category",
    "Card_Category"
]

QUANTIATIVE_COLS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio"
]

FEATURES_TO_KEEP = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn'
]

def import_data(path):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(path)
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return df


def perform_eda(df, output_path="./images/"):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            output_path: directory to write images

    output:
            None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"DataFrame Shape: {df.shape}\n")
    print(f"DataFrame Null Counts:\n {df.isnull().sum()}\n")
    print(f"DataFrame Description:\n {df.describe()}\n")

    print("Saving Churn Histogram")
    df["Churn"].hist(figsize=(20,10))
    plt.savefig(os.path.join(output_path,"churn_histogram.png"))

    print("Saving Cusomer Age Histogram")
    df["Customer_Age"].hist(figsize=(20,10))
    plt.savefig(os.path.join(output_path,"customer_age_histogram.png"))

    print("Saving Marital Status Bar Chart")
    df.Marital_Status.value_counts("normalize").plot(
        kind="bar",figsize=(20,10)
    )
    plt.savefig(os.path.join(output_path,"marital_status_bar.png"))

    print("Saving Total Trans COunt Density Chart")
    plt.figure(figsize=(20,10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join(output_path,"total_trans_count_density.png"))

    print("Saving Correlation Heatmap")
    plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths = 2)
    plt.savefig(os.path.join(output_path,"correlation_heatmap.png"))


def encoder_helper(df, category_lst, response='Churn'):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    try:
        assert isinstance(df, pd.DataFrame)
        assert isinstance(category_lst, list)
        assert isinstance(response, str)

        for category_col in category_lst:
            col_group_dict = df.groupby(category_col)[response].mean().to_dict()
            df[category_col+'_'+response] = df[category_col].map(col_group_dict)

        return df

    except AssertionError as err:
        print("""Inputs of wrong type.
                 Expected pd.DataFrame, List, str[Optional]""")
        raise err
    except KeyError as err:
        print("Variable in category_lst or response not in Data Frame")
        raise err


def perform_feature_engineering(df, response='Churn'):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    try:
        assert isinstance(df, pd.DataFrame)
        assert isinstance(response, str)

        df = encoder_helper(df, CATEGORICAL_COLS)

        y = df[response]
        X = df[FEATURES_TO_KEEP]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size= 0.3, random_state=42
        )

        return X_train, X_test, y_train, y_test

    except AssertionError as err:
        print("""Inputs of wrong type.
                 Expected pd.DataFrame, str[Optional]""")
        raise err

    except KeyError as err:
        print("Expected columns not in Data Frame")
        raise err

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_path="./images/"):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig(os.path.join(output_path, 'randomforest_report.png'))

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig(os.path.join(output_path, 'logisticregression_report.png'))


def feature_importance_plot(model, X_data, output_path):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_path: path to store the figure

    output:
             None
    """
    try:
        importances = model.feature_importances_
    except AttributeError as err:
        print("Model doesn't have feature importances")
        raise err

    try:
        assert isinstance(X_data, pd.DataFrame)
    except AssertionError as err:
        print('X_data should be pandas DataFrame')
        raise err

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(os.path.join(output_path, 'feature_importance.png'))


def train_models(X_train, X_test, y_train, y_test, model_path="./models", image_path="./images"):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              model_path: path for storing models
              image_path: path for storing images
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(os.path.join(image_path, 'model_roc.png'))

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_path=image_path)

    feature_importance_plot(cv_rfc.best_estimator_,
                            X_test,
                            output_path=image_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    joblib.dump(cv_rfc.best_estimator_,
                os.path.join(model_path,'rfc_model.pkl'))
    joblib.dump(lrc,
                os.path.join(model_path,'logistic_model.pkl'))

if __name__ == "__main__":
    df = import_data("data/bank_data.csv")
    perform_eda(df)
    df = encoder_helper(df, CATEGORICAL_COLS)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df.head(1000))
    train_models(X_train, X_test, y_train, y_test)
