import pandas as pd
import numpy as np
from random import randrange

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc


def stplot(fig):
    return st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False},)

def sttable(df):
    return st.dataframe(df, use_container_width=True, hide_index= True)
###

st.title('Employee Retention Prediction')

# LOAD DATA
FILEPATH = './HR_Analytics.csv'
df = pd.read_csv(FILEPATH)
new_col_names = {
    'average_montly_hours': 'average_monthly_hours',
    'time_spend_company': 'tenure',
    'Work_accident': 'work_accident',
    'Department': 'department'
}
df.rename(columns=new_col_names, inplace=True)
df = df.drop_duplicates()

###
st.divider()

st.header('Business Problem')
st.write('''
    Employee recruitment is time and cost consuming. As such, a tool to predict and identify motives behind an employee leaving 
    (resignation or laid off) can be extremely beneficial to the company and employee satisfaction. The goal of this project is 
    to build a model that can predict whether an employee will leave the company and suggest reasons why to the human resources
    department.
''')

###
st.divider()

st.header('Data')
# DATA SCHEMA
url = 'https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv'
st.write('The data was obtained from the [HR Analytics Job Prediction](%s) dataset by Faisal Qureshi on Kaggle.' %url)
tagDict = pd.DataFrame({
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
sttable(tagDict)
st.caption('*Note some variable names have been changed for uniformality.')

###

st.subheader('Exploratory Data Analysis')
st.write('''
    1. Renamed Columns
    2. Checked for nulls → none were found
    3. Checked for duplicates → 3008 rows were found \\
    Duplicate rows were dropped as they were likely the same employee recorded multiple times.
    4. Checked for outliers in tenure → 824 found (see figure below)\\
    Outliers were kept since the final model uses a random forest which is less sensitive to outliers.
    5. Encoded categorical features: department and salary.
''')

tenureBoxplot = px.box(
    df, 
    y=['tenure'],
    title = 'Tenure Distribution',
    color_discrete_sequence = ['#636EFA'],
)
tenureBoxplot.update_layout(
    margin = dict(l=20, r=20, t=40, b=0),
    height = 300,
    xaxis = dict(
        visible = False
    ),
    yaxis = dict(
        title = 'Years'
    ),
)
stplot(tenureBoxplot)

###

st.subheader('Visualizing')
# PEARSONS CORRELATION
df_corr = df.corr()
mask = np.triu(np.ones_like(df_corr, dtype=bool))
corr_mask = df_corr.mask(mask)
pearCorr = go.Figure()
pearCorr.add_trace(
    go.Heatmap(
        x = corr_mask.columns.tolist(),
        y = corr_mask.columns.tolist(),
        z = corr_mask.to_numpy(),
        texttemplate = '%{z:.2f}',
        autocolorscale = False,
        colorscale = ['#636EFA', '#FFCCEA'],
        # hoverinfo="none",
    )
)
pearCorr.update_layout(
    title = 'Pearson Correlations Between Features',
    margin = dict(l=20, r=20, t=40, b=0),
    height = 400,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis_autorange='reversed',
)
stplot(pearCorr)

st.write('Overall, there appears to be few linear relationships, but some correlations can be seen between:')
col1, col2 = st.columns(2)
with col1:
    st.write('''
        - left and satisfaction level
        - average monthly hours and last evaluation
    ''')
with col2:
    st.write('''
        - number projects and last evaluation
        - average monthly hours and number project
    ''')
st.write('')

summary = pd.DataFrame({
    'Left': [1, 0],
    'Count': df.groupby('left').size(),
    'Mean Satisfaction': df.groupby('left').mean().satisfaction_level,
    'Mean Monthly Hours': df.groupby('left').mean().average_monthly_hours,
    'Mean Last Evaluation': df.groupby('left').mean().last_evaluation,
    'Mean Number Projects': df.groupby('left').mean().number_project,
})
sttable(summary)
st.write('''
    The dataset is imbalanced with roughly 83% of employees having stayed and the minority 17% having left the company.
    Additionally, as expected, employees who left were on average more dissatisfied than their counterparts.
''')
st.write('')

col1, col2 = st.columns(2)
with col1:
    left = df[df.left==1].groupby('number_project').mean().average_monthly_hours.rename('1')
    stay = df[df.left==0].groupby('number_project').mean().average_monthly_hours.rename('0')
    dfTmp = pd.concat((stay, left), axis=1, join='outer')
    dfTmp.reset_index(inplace=True)
    numProj_aveHours = px.bar(
        dfTmp, 
        x = 'number_project', 
        y = ['0', '1'], 
        barmode = 'group',
        title = 'Average Monthly Hours per Number Projects',
        color_discrete_sequence=['#636EFA', '#FF6692'],
    )
    numProj_aveHours.update_layout(
        margin = dict(l=20, r=50, t=30, b=0),
        xaxis = dict(
            title = 'Number of Projects',
            dtick = 1,
        ),
        yaxis = dict(
            title = 'Average Monthly Hours'
        ),
        legend_title = 'Left',
        # showlegend = False,
        height = 300,
        
    )
    stplot(numProj_aveHours)
with col2:
    monthlyHours = px.histogram(
        df, 
        x = 'average_monthly_hours', 
        color = 'left', 
        color_discrete_sequence = ['#FF6692', '#636EFA'],
        histnorm = 'percent',
        nbins = 30,
        title = 'Distribution of Average Monthly Hours',
    )
    monthlyHours.update_layout(
        hovermode='x unified', 
        barmode='overlay',
        height = 300,
        xaxis = dict(
            title = 'Average Monthly Hours',
        ),
        yaxis = dict(
            title = 'Percent Employees',
        ),
        margin = dict(l=20, r=0, t=30, b=0),
        legend = {'traceorder':'reversed'},
        legend_title = 'Left',
    )
    monthlyHours.update_traces(opacity=0.75)
    monthlyHours.add_vline(
        x = df.average_monthly_hours.mean(),
        line_width = 1, 
        line_dash = 'dash', 
        annotation_text = f'total mean = {int(df.average_monthly_hours.median())}',
    )
    stplot(monthlyHours)
st.write('''
    Employees that left appear to fall into one of two categories. 
    1. Were involved in two projects and worked very few hours a month compared to their colleagues. These employees were likely 
       terminated and had their workload decreased or resigned and were handing their work over to a replacement.
    2. Were involved in three or more projects and tended to work more hours than their peers. Notably, employees on seven projects 
       had worked on average 60 hours a week, greater than any other group, and have all left the company.
''')
st.write('')

promote = px.scatter(
    title = 'Satisfaction Corresponding to Worked Monthly Hours'
)
promote.add_trace(
    go.Scatter(
        x = df[df.left==0].satisfaction_level, 
        y = df[df.left==0].average_monthly_hours,
        mode = 'markers',
        marker_color = '#636EFA',
        marker_opacity = 0.75,
        name = '0'
    )
)
promote.add_trace(
    go.Scatter(
        x = df[df.left==1].satisfaction_level, 
        y = df[df.left==1].average_monthly_hours,
        mode = 'markers',
        marker_color = '#FF6692',
        marker_opacity = 0.75,
        name = '1'
    )
)
promote.update_layout( 
    height = 300,
    margin = dict(l=20, r=0, t=30, b=0),
    xaxis = dict(
        title = 'Satisfaction Level',
    ),
    yaxis = dict(
        title = 'Average Monthly Hours'
    ),
    legend_title = 'Left',
)
stplot(promote)
st.write(''' 
    The diagram shows three interesting clusters regarding employees who left:
    1. All employees that reported satisfaction levels below 0.12 left the company, with most of them working extreme amounts
       of hours (54-70 hours a week). They likely left because of overwork.
    2. Employees with satisfaction scores between 0.36 and 0.46 and worked below 40 hours a week, also left. This is likely
       the group that was terminated or were handing work over.
    3. A third group of employees that also worked a significant amount of time (48-62 hours a week) reported high satisfactions
       between 0.72 and 0.93. This group may have also felt overworked or have found better opportunities elsewhere, but there 
       is no clear motive for them leaving the company.
''')
st.write('')

left_promo = df[(df.left==1)&(df.promotion_last_5years==1)].groupby('tenure').size().rename('Promoted and Left')
left_nopromo = df[(df.left==1)&(df.promotion_last_5years==0)].groupby('tenure').size().rename("Wasn't Promoted and Left")
stay_promo = df[(df.left==0)&(df.promotion_last_5years==1)].groupby('tenure').size().rename('Promoted and Stayed')
stay_nopromo = df[(df.left==0)&(df.promotion_last_5years==0)].groupby('tenure').size().rename("Wasn't Promoted and Stayed")
left = pd.concat((left_nopromo, left_promo), axis=1, join='outer')
stay = pd.concat((stay_nopromo, stay_promo), axis=1, join='outer')
dfTmp = pd.concat((stay, left), axis=1, join='outer')
tenure_promo = px.bar(
    dfTmp,
    color_discrete_sequence = ['#FF6692', '#FFCCEA', '#AB63FA', '#636EFA'],
    title = 'Promotions by Tenure',
)
tenure_promo.update_layout( 
    hovermode = 'x unified',
    height = 300,
    margin = dict(l=20, r=0, t=30, b=0),
    xaxis = dict(
        title = 'Tenure',
    ),
    yaxis = dict(
        title = 'Number of Employees'
    ),
    legend_title = '',
)
stplot(tenure_promo)
st.write('''
    Tenure did not have a huge effect on promotions, in fact most promotion were given to employees who had been working for two years.
    Despite the lack of promotions for more senior employees however, all employees that made it past six years with the company have
    not left and the amount of employees that leave decrease the longer they've been with the company.
''')
st.write('')

col1, col2 = st.columns(2)
with col1:
    left = df[df.left==1].groupby('number_project').size().rename('1')
    stay = df[df.left==0].groupby('number_project').size().rename('0')
    dfTmp = pd.concat((stay, left), axis=1, join='outer')
    dfTmp.reset_index(inplace=True)
    numProj = px.bar(
        dfTmp, 
        x = 'number_project', 
        y = ['0', '1'], 
        barmode = 'group',
        title = 'Number of Projects Assigned to Employees',
        color_discrete_sequence = ['#636EFA', '#FF6692'],
    )
    numProj.update_layout(
        margin = dict(l=20, r=0, t=30, b=0),
        xaxis = dict(
            title = 'Number of Projects',
            dtick = 1,
        ),
        yaxis = dict(
            title = 'Number of Employees'
        ),
        legend_title = 'Left',
        height = 300
    )
    stplot(numProj)
with col2:
    show2 = df[df.number_project==2]
    show6 = df[df.number_project==6]
    hide = df[(df.number_project!=2)&(df.number_project!=6)].sort_values('number_project')
    numPro_lastEval = px.histogram(
        show2, 
        x = 'last_evaluation', 
        color = 'number_project', 
        color_discrete_sequence = ['#636EFA'],
        histnorm = 'percent', 
        nbins = 30, 
        title = 'Evaluation Scores by Number of Projects',
        height = 300,
    )
    hide = px.histogram(
        hide, 
        x = 'last_evaluation', 
        color = 'number_project', 
        color_discrete_sequence = ['#EF553B', '#00CC96', '#AB63FA', '#FECB52'],
        histnorm = 'percent', 
        nbins = 20,
    )
    hide.update_traces(visible='legendonly')
    for trace in hide.data[0:3]:
        numPro_lastEval.add_trace(trace)
    show6 = px.histogram(
        show6, 
        x = 'last_evaluation', 
        color = 'number_project', 
        color_discrete_sequence = ['#FF6692'],
        histnorm = 'percent', 
        nbins=20, 
    )
    numPro_lastEval.add_trace(show6.data[0])
    numPro_lastEval.add_trace(hide.data[-1])
    numPro_lastEval.update_layout(
        hovermode = 'x unified', 
        barmode = 'overlay', 
        legend = {'traceorder':'normal'},
        margin = dict(l=20, r=0, t=30, b=0),
        xaxis = dict(
            title = 'Last Evaluation',
        ),
        yaxis = dict(
            title = 'Percent Employees'
        ),
    )
    numPro_lastEval.update_traces(opacity=0.75)
    stplot(numPro_lastEval)
st.write('''
    As observed earlier, a large number of employees that were on two projects left and all employees on seven left. 
    But beginning at three projects, as the number of projects increased, the number of employees that left also
    increased. At six projects, the number of employees that stayed and left were somewhat similar. 
    
    Looking closer into the two and six project groups, employees who contributed to more projects tended to be better 
    evaluated. By toggling on seven projects in the plot, it is clear most of these employees were highly valuable to 
    their teams, making it a even greater loss that many of them left. Low evaluations for the two project group also
    suggests that many of them may have been terminated.
''')

###

st.subheader('Insights')
st.write('''
    Issues in employee retention can be tied to long work hours, overloading employees' workloads with projects, and 
    disatisfaction in their work. 

    To summarize:
    - The dataset is imbalanced with a greater number of employees that have not left the company.
    - Employees that leave are on average, more dissatisfied.
    - Most employees who left were a part of one of two categories:
        - Had two projects and worked little hours. They likely left due to termination or were handing over responsibilities.
        - Were on five to seven projects and worked considerably more hours than their peers. They likely left due to 
        burnout or to pursue better positions.
    - The likelihood of an employee leaving decreases the longer they've been with the company.
    - Employees with more projects are more positively evaluated.
''')

###
st.divider()

st.header('Model')
st.write('''
    - The Pearson correlations indicate very few linear relationships therefore regression may not be the best choice for a model
    - There is a lot of data (11991 rows) compared to a few features (10) so the data would be prone to underfitting. As such, regression
      may not be the best model, and a decision tree may be a better starting point.
    - Because the dataset is imbalanced, use of accuracy as the evaluating metric should be avoided. I'll be using F1 as the main evaluation metric.
    - The data was split into 80% training and 20% validation.
    - Optimal hyperparameters were found using grid searches.
''')

df = pd.get_dummies(df)
y = df.left
X = df.drop('left', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

st.subheader('Decision Tree') 
code = '''
    tree = DecisionTreeClassifier(
        max_depth = 5,
        max_features = 1.0,
        min_samples_leaf = 2,
        min_samples_split = 4,
    )
'''
st.code(code, language='python')

tree = DecisionTreeClassifier(
    max_depth = 5,
    max_features = 1.0,
    min_samples_leaf = 2,
    min_samples_split = 4,
)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

col1, col2 = st.columns(2)
with col1:
    tree_report = pd.DataFrame({
        '': ['0', '1', 'Accuracy', 'Macro Avg',],# 'Weighted Avg'],
        'Precision': ['0.99', '0.92', '', '0.95',],#  '0.98'],
        'Recall': ['0.98', '0.93', '', '0.96',],#  '0.97'],
        'F1': ['0.99', '0.92', '0.97', '0.95',],#  '0.98'],
        'Support': ['2009', '390', '2399', '2399',],#  '2399']
    })
    sttable(tree_report)
    tree_metrics = pd.DataFrame({
        '': ['Training', 'Validation'],
        'Precision': ['96.85', '91.88'],
        'Recall': ['92.25', '92.82'],
        'F1': ['94.50', '92.35'],
        'Accuracy': ['98.21', '97.50'],
    })
    sttable(tree_metrics)
with col2:
    tree_cm = confusion_matrix(y_test, y_pred_tree, labels=tree.classes_)
    tree_confusion_matrix = go.Figure()
    tree_confusion_matrix.add_trace(
        go.Heatmap(
            z = tree_cm, 
            x = ['0', '1'], 
            y = ['0', '1'], 
            texttemplate = '%{z:f}',
            autocolorscale = False,
            colorscale = ['#636EFA', '#FFCCEA'],
        ),
    )
    tree_confusion_matrix.update_layout(
        height = 310,
        title = 'Random Forest Confusion Matrix',
        margin = dict(l=0, r=10, t=30, b=0),
    )
    tree_confusion_matrix.update_xaxes(
        title = 'Predicted',
        type = 'category',
    )
    tree_confusion_matrix.update_yaxes(
        title = 'True',
        type = 'category',
    )
    stplot(tree_confusion_matrix)
st.write('''
    The decision tree performed quite well and due to the imbalance in data, it is expected for the model to have more difficulty correctly 
    identifying employees who leave, as evident in the lower metric scores for left=1. While it is a good first attempt, it appears 
    to be slightly overfitting. Since the training precision is 5% higher than validation precision, it is observed that the model 
    has difficulty identifying true from false positives. While a 5% difference is small, it can be seen in the confusion matrix that 
    32 employees were incorrectly predicted as likely to leave and 28 were incorrectly predicted as likely to stay.

    Since decision trees are sensitive to overfitting, the next step would be to test a random forest model (uses an ensemble of 
    decision trees to avoid overfitting).
''')

st.subheader('Random Forest')
code = '''
    rf = RandomForestClassifier(
        max_depth = None,
        max_features = 1.0,
        max_samples = 0.7,
        min_samples_leaf = 2,
        min_samples_split = 2,
        n_estimators = 100
    )
'''
st.code(code, language='python')

rf = RandomForestClassifier(
    max_depth = None,
    max_features = 1.0,
    max_samples = 0.7,
    min_samples_leaf = 2,
    min_samples_split = 2,
    n_estimators = 100
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

col1, col2 = st.columns(2)
with col1:
    rf_report = pd.DataFrame({
        '': ['0', '1', 'Accuracy', 'Macro Avg',],# 'Weighted Avg'],
        'Precision': ['0.99', '0.98', '', '0.98',],#  '0.98'],
        'Recall': ['1.00', '0.92', '', '0.96',],#  '0.98'],
        'F1': ['0.99', '0.95', '0.98', '0.97',],#  '0.98'],
        'Support': ['2009', '390', '2399', '2399',],#  '2399']
    })
    sttable(rf_report)
    rf_metrics = pd.DataFrame({
        '': ['Training', 'Validation'],
        'Precision': ['99.20', '98.09'],
        'Recall': ['92.44', '92.31'],
        'F1': ['95.70', '95.11'],
        'Accuracy': ['98.61', '98.46'],
    })
    sttable(rf_metrics)
with col2:
    rf_cm = confusion_matrix(y_test, y_pred_rf, labels=rf.classes_)
    rf_confusion_matrix = go.Figure()
    rf_confusion_matrix.add_trace(
        go.Heatmap(
            z = rf_cm, 
            x = ['0', '1'], 
            y = ['0', '1'], 
            texttemplate = '%{z:.2f}',
            autocolorscale = False,
            colorscale = ['#636EFA', '#FFCCEA'],
        ),
    )
    rf_confusion_matrix.update_layout(
        height = 310,
        title = 'Random Forest Confusion Matrix',
        margin = dict(l=0, r=10, t=30, b=0),
    )
    rf_confusion_matrix.update_xaxes(
        title = 'Predicted',
        type = 'category',
    )
    rf_confusion_matrix.update_yaxes(
        title = 'True',
        type = 'category',
    )
    stplot(rf_confusion_matrix)
st.write('''
    The random forest performed very well despite the imbalance in data. Additionally, training and validation metric scores are 
    very close indicating that the model isn't overfitting and the model predicted significantly less false positives compared to 
    the decision tree.
''')
rf_fpr, rf_tpr, _ = roc_curve(y_test, y_pred_rf)
rf_roc = px.area(
    x = rf_fpr, 
    y = rf_tpr,
    title = f'Random Forest ROC Curve (AUC={auc(rf_fpr, rf_tpr):.4f})',
    labels = dict(
        x = 'False Positive Rate', 
        y = 'True Positive Rate',
    ),
    color_discrete_sequence = ['#636EFA'],
)
rf_roc.add_shape(
    type = 'line', 
    line = dict(
        dash = 'dash',
        color = '#636EFA',
    ),
    x0 = 0, x1 = 1, 
    y0 = 0, y1 = 1,
)
rf_roc.update_layout(
    margin = dict(l=10, r=10, t=30, b=0),
    height = 300,
)
stplot(rf_roc)

st.subheader('Evaluation')
eval_metrics = pd.DataFrame({
    '': ['Decision Tree', 'Random Forest'],
    'Precision': ['91.88', '98.09'],
    'Recall': ['92.82', '92.31'],
    'F1': ['92.35', '95.11'],
    'Accuracy': ['97.50', '98.46'],
})
sttable(eval_metrics)
st.write('''
    Precision and F1 scores in the random forest were considerably better than the decision tree and, as seen previously, the random 
    forest had similar false negatives but much less false positives. Thus the random forest is the slightly better model for this application.
''')
st.write('')

importance = pd.DataFrame({
    'Feature': [
        'satisfaction_level', 
        'number_project', 
        'last_evaluation', 
        'tenure', 
        'average_monthly_hours', 
        'salary_low', 
        'department_sales', 
        'department_technical', 
        'salary_medium', 
        'department_support'
    ],
    'Importance': [
        0.4656167,
        0.1558797,
        0.154371,
        0.1160692,
        0.09387437,
        0.002243415,
        0.002119278,
        0.001976613,
        0.001680455,
        0.001085423,
    ],
})
feature_importance = px.bar(
    importance, 
    x='Feature', y='Importance',
    title = 'Random Forest Feature Importance',
    color_discrete_sequence = ['#636EFA'],
)
feature_importance.update_layout(
    margin = dict(l=10, r=10, t=30, b=0),
    height = 300,
)
stplot(feature_importance)
st.write('''
    Additionally, it is seen that satisfaction level plays the biggest role in determining whether an employee will 
    leave. Last evaluation, number of projects, tenure, and average monthly hours also play a role.
''')

###
st.divider()

st.header('Conclusion')
st.write('''
    Many of the issues surrounding employee retention stem from excessive overtime, an overload in projects, and 
    disatisfaction. 
''')
st.subheader('Recommendations')
st.write('''
    - More explicit guidelines surrounding overtime and conversations regarding time off and mental health.
    - Better distribution of projects and communication with employees regarding their ability to complete
      their tasks.
        - Implement a maximum project number of four or five.
    - Review evaluation score metrics, are they fair to all employees or do they favor employees that work more
      projects and hours. Quantity does not always equate to quality.
''')

st.subheader('Next Steps')
st.write('''
    - Consider feature engineering, since the models' scores were very high, data leakage is likely.
    - Evaluate whether gradient boosting would improve performance.
''')

###
st.divider()

st.header('Extra')
st.write('''
    Here is small UI to predict whether an employee is likely to leave the company based on the random forest model.
    Feel free to play with the values and see what the model would predict.
''')

departments = {
    'IT': 'department_IT',	
    'RandD': 'department_RandD',
    'Accounting': 'department_accounting',
    'HR': 'department_hr',
    'Management': 'department_management',
    'Marketing': 'department_marketing',
    'Product Management': 'department_product_mng',
    'Sales': 'department_sales',
    'Support': 'department_support',
    'Technical': 'department_technical',
}	
salaries = {
    'Low': 'salary_low',
    'Medium': 'salary_medium',
    'High': 'salary_high',
}
col1, col2, col3, col4= st.columns(4)
with col1:
    sat_lvl = st.number_input('Satisfaction Level', min_value=0.0, max_value=1.0, value=0.5)
    tenure = st.number_input('Tenure', min_value=0, max_value=100, value=2)
with col2:
    last_eval = st.number_input('Last Evaluation', min_value=0.0, max_value=1.0, value=0.5)
    dept = st.selectbox('Department', departments.keys())
with col3:
    num_proj = st.number_input('Number Projects', min_value=2, max_value=7, value=4)
    sal = st.selectbox('Salary', salaries.keys(), index=1)
with col4:
    ave_hours = st.number_input('Average Monthly Hours', min_value=0, max_value=744, value=180)
    work_acc = st.checkbox('Work Accident')
    promo = st.checkbox('Promotion in the Last 5 Years')

user_df = pd.DataFrame({
    'satisfaction_level': sat_lvl,
    'last_evaluation': last_eval,
    'number_project': num_proj, 
    'average_monthly_hours': ave_hours,
    'tenure': tenure,
    'work_accident': int(work_acc),
    'promotion_last_5years': int(promo),
    'department_IT': 0,
    'department_RandD': 0,
    'department_accounting': 0,
    'department_hr': 0,
    'department_management': 0, 
    'department_marketing': 0,
    'department_product_mng': 0,
    'department_sales': 0,
    'department_support': 0,
    'department_technical': 0,
    'salary_high': 0,
    'salary_low': 0,
    'salary_medium': 0,
}, index=[0])
user_df[departments[dept]] = 1
user_df[salaries[sal]] = 1

prediction = rf.predict(user_df)
if prediction == 1:
    st.info('The employee is likely to leave the company.')
elif prediction == 0:
    st.info('The employee is likely to continue working at the company.')

###
st.divider()

st.header('Endnote')
url = 'https://www.linkedin.com/in/ying-sunwan/'
st.write('''
    Thank you for viewing my project! Please check my [LinkedIn](%s) for more.\\
    This project was completed as part of the Google Advanced Data Analytics Certification capstone.
''' %url)
