import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score

def st_plot(fig):
    '''
    Plots a Plotly diagram to Streamlit with formatting.

    Args:
        fig (Plotly figure)
    Returns:
        Streamlit plotly_chart 
    '''
    return st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False},)

def st_table(df):
    '''
    Writes a dataframe to Streamlit with formatting.

    Args:
        df (DataFrame)
    Returns:
        Streamlit dataframe
    '''
    return st.dataframe(df, use_container_width=True, hide_index= True)

@st.cache_data
def load_data(filepath):
    '''
    Loads the .csv file at filepath, renames columns, and drops duplicates.

    Args:
        filepath (string)
    Returns:
        df (DataFrame): Cleaned DataFrame of .csv data.
    '''
    df = pd.read_csv(filepath)
    new_col_names = {
        'average_montly_hours': 'average_monthly_hours',
        'time_spend_company': 'tenure',
        'Work_accident': 'work_accident',
        'Department': 'department'
    }
    df.rename(columns=new_col_names, inplace=True)
    df = df.drop_duplicates()
    return df

def get_tag_dict():
    '''
    Returns the tag dictionary for the HR dataset.

    Returns:
        tag_dict (DataFrame)
    '''
    tag_dict = pd.DataFrame({
        'Variable': [
            'satisfaction_level (float)', 
            'last_evaluation (float)',
            'number_project (integer)',
            'average_monthly_hours* (integer)',
            'tenure* (integer)',
            'work_accident* (boolean)',
            'left (boolean)',
            'promotion_last_5years (boolean)',
            'department* (string)',
            'salary (string)',
        ],
        'Description': [
            'Employee-reported job satisfaction level from 0 to 1, inclusive.',
            'Employee’s last performance review score from 0 to 1, inclusive.',
            'Number of projects employee contributes to.',
            'Average number of hours employee worked per month.',
            'Number of years the employee has been with the company.',
            'Whether or not the employee experienced an accident while at work.',
            'Whether or not the employee left the company.',
            'Whether or not the employee was promoted in the last 5 years.',
            'The employee’s department.',
            'The employee’s salary category (low, medium, high).',
        ]
    })
    return tag_dict

def get_summary_stats(df):
    '''
    Calculates size, mean satisfaction level, mean monthly hours, mean last evaluation, and mean number
    of projects for employees that left and have not left.

    Args:
        df (DataFrame)
    Returns:
        summary (DataFrame): Summary statistics for employees that left and have not left.
    '''
    summary = pd.DataFrame({
        'Left': [1, 0],
        'Count': df.groupby('left').size(),
        'Mean Satisfaction': df.groupby('left').mean().satisfaction_level,
        'Mean Monthly Hours': df.groupby('left').mean().average_monthly_hours,
        'Mean Last Evaluation': df.groupby('left').mean().last_evaluation,
        'Mean Number Projects': df.groupby('left').mean().number_project,
    })
    return summary

def split_data(df):
    '''
    Splits the input data for model training. Target is the left column while everything else
    is a feature.

    Args:
        df (DataFrame)
    Returns:
        df_dummies (DataFrame): Input df with label encoding.
        X_train (DataFrame): 80% of feature data.
        X_test (DataFrame): 20% of feature data reserved for model validation.
        y_train (DataFrame): 80% of taget data.
        y_test (DataFrame): 20% of target data reserved for model validation.
    '''
    df_dummies = pd.get_dummies(df)
    y = df_dummies.left
    X = df_dummies.drop('left', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    return df_dummies, X_train, X_test, y_train, y_test

@st.cache_data
def get_dt(df, X_train, X_test, y_train):
    '''
    Trains, fits, and predicts a decision tree model.

    Args:
        df (DataFrame)
        X_train (DataFrame)
        X_test (DataFrame)
        y_train (DataFrame)
    Returns:
        tree (DecisionTreeClassifier)
        y_pred (DataFrame)
    '''
    tree = DecisionTreeClassifier(
        max_depth = 5,
        max_features = 1.0,
        min_samples_leaf = 2,
        min_samples_split = 4,
    )
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    return tree, y_pred

def get_tree_report():
    report = pd.DataFrame({
        '': ['0', '1', 'Accuracy', 'Macro Avg',],# 'Weighted Avg'],
        'Precision': ['0.99', '0.92', '', '0.95',],#  '0.98'],
        'Recall': ['0.98', '0.93', '', '0.96',],#  '0.97'],
        'F1': ['0.99', '0.92', '0.97', '0.95',],#  '0.98'],
        'Support': ['2009', '390', '2399', '2399',],#  '2399']
    })
    return report

def get_tree_metrics():
    metrics = pd.DataFrame({
        '': ['Training', 'Validation'],
        'Precision': ['96.85', '91.88'],
        'Recall': ['92.25', '92.82'],
        'F1': ['94.50', '92.35'],
        'Accuracy': ['98.21', '97.50'],
    })
    return metrics

@st.cache_data
def get_rf(df, X_train, X_test, y_train):
    '''
    Trains, fits, and predicts a random forest model.

    Args:
        df (DataFrame)
        X_train (DataFrame)
        X_test (DataFrame)
        y_train (DataFrame)
    Returns:
        tree (RandomForestClassifier)
        y_pred (DataFrame)
    '''
    rf = RandomForestClassifier(
        max_depth = None,
        max_features = 1.0,
        max_samples = 0.7,
        min_samples_leaf = 2,
        min_samples_split = 2,
        n_estimators = 100
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return rf, y_pred

def get_rf_report():
    report = pd.DataFrame({
        '': ['0', '1', 'Accuracy', 'Macro Avg',],# 'Weighted Avg'],
        'Precision': ['0.99', '0.98', '', '0.98',],#  '0.98'],
        'Recall': ['1.00', '0.92', '', '0.96',],#  '0.98'],
        'F1': ['0.99', '0.95', '0.98', '0.97',],#  '0.98'],
        'Support': ['2009', '390', '2399', '2399',],#  '2399']
    })
    return report

def get_rf_metrics():
    metrics = pd.DataFrame({
        '': ['Training', 'Validation'],
        'Precision': ['99.20', '98.09'],
        'Recall': ['92.44', '92.31'],
        'F1': ['95.70', '95.11'],
        'Accuracy': ['98.61', '98.46'],
    })
    return metrics

def get_eval_metrics():
    eval_metrics = pd.DataFrame({
        '': ['Decision Tree', 'Random Forest'],
        'Precision': ['91.88', '98.09'],
        'Recall': ['92.82', '92.31'],
        'F1': ['92.35', '95.11'],
        'Accuracy': ['97.50', '98.46'],
    })
    return eval_metrics