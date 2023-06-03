import pandas as pd
import numpy as np

import streamlit as st

from src.util import *
from src.visualizations import *
from src import predictor

st.title('Employee Retention Prediction')

FILEPATH = './src/HR_Analytics.csv'
df = load_data(FILEPATH)

st.divider() #--------------------------------------------------------------------------------------------------------

st.header('Business Problem')
st.write('''
    Employee recruitment is time and cost consuming. As such, a tool to predict and identify motives behind an
    employee leaving (resignation or laid off) can be extremely beneficial to the company and employee satisfaction. 
    The goal of this project is to build a model that can predict whether an employee will leave the company and 
    suggest reasons.
''')

st.divider() #--------------------------------------------------------------------------------------------------------

st.header('Data')
kaggle_url = 'https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv'
st.write('The data was obtained from the [HR Analytics Job Prediction](%s) dataset by Faisal Qureshi on Kaggle.' %kaggle_url)
st_table(get_tag_dict())
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
st_plot(plot_tenure_boxplot(df))

###

st.subheader('Visualizing')
st_plot(plot_correlations(df))
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

st_table(get_summary_stats(df))
st.write('''
    The dataset is imbalanced with roughly 83% of employees having stayed and the minority 17% having left the 
    company. Additionally, as expected, employees who left were on average more dissatisfied than their counterparts.
''')
st.write('')

col1, col2 = st.columns(2)
with col1:
    st_plot(plot_hours_per_project(df))
with col2:
    st_plot(plot_hours_hist(df))

st.write('''
    Employees that left appear to fall into one of two categories. 
    1. Were involved in two projects and worked very few hours a month compared to their colleagues. These employees 
       were likely terminated and had their workload decreased or resigned and were handing their work over to a 
       replacement.
    2. Were involved in three or more projects and tended to work more hours than their peers. Notably, employees on 
       seven projects had worked on average 60 hours a week, greater than any other group, and have all left the 
       company.
''')
st.write('')

st_plot(plot_satisfaction(df))
st.write(''' 
    The diagram shows three interesting clusters regarding employees who left:
    1. All employees that reported satisfaction levels below 0.12 left the company, with most of them working extreme
       amounts of hours (54-70 hours a week). They likely left because of overwork.
    2. Employees with satisfaction scores between 0.36 and 0.46 and worked below 40 hours a week, also left. This is
       likely the group that was terminated or were handing work over.
    3. A third group of employees that also worked a significant amount of time (48-62 hours a week) reported high 
       satisfactions between 0.72 and 0.93. This group may have also felt overworked or have found better opportunities 
       elsewhere, but there is no clear motive for them leaving the company.
''')
st.write('')

st_plot(plot_tenure_promotions(df))
st.write('''
    Tenure did not have a huge effect on promotions, in fact most promotion were given to employees who had been 
    working for two years. Despite the lack of promotions for more senior employees however, all employees that made 
    it past six years with the company have not left and the amount of employees that leave decrease the longer they've 
    been with the company.
''')
st.write('')

col1, col2 = st.columns(2)
with col1:
    st_plot(plot_num_projects(df))
with col2:
    st_plot(plot_eval_per_project(df))
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
        - Had two projects and worked little hours. They likely left due to termination or were handing over 
          responsibilities.
        - Were on five to seven projects and worked considerably more hours than their peers. They likely left due to 
        burnout or to pursue better positions.
    - The likelihood of an employee leaving decreases the longer they've been with the company.
    - Employees with more projects are more positively evaluated.
''')

st.divider() #--------------------------------------------------------------------------------------------------------

st.header('Model')
st.write('''
    - The Pearson correlations indicate very few linear relationships therefore regression may not be the best choice 
      for a model
    - There is a lot of data (11991 rows) compared to a few features (10) so the data would be prone to underfitting. 
      As such, regression may not be the best model, and a decision tree may be a better starting point.
    - Because the dataset is imbalanced, use of accuracy as the evaluating metric should be avoided. I'll be using F1 
      as the main evaluation metric.
    - The data was split into 80% training and 20% validation.
    - Optimal hyperparameters were found using grid searches and 5-fold cross validation.
''')
df_dummies, X_train, X_test, y_train, y_test = split_data(df)

###

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
dt, y_pred_dt = get_dt(df_dummies, X_train, X_test, y_train)
col1, col2 = st.columns(2)
with col1:
    st_table(get_tree_report())
    st_table(get_tree_metrics())
with col2:
    st_plot(plot_confusion_matrix(y_test, y_pred_dt, dt, 'Decision Tree Confusion Matrix'))

st.write('''
    The decision tree performed quite well and due to the imbalance in data, it is expected for the model to have more 
    difficulty correctly identifying employees who leave, as evident in the lower metric scores for left=1. While it 
    is a good first attempt, it appears to be slightly overfitting. Since the training precision is 5% higher than 
    validation precision, it is observed that the model has difficulty identifying true from false positives. While a 
    5% difference is small, it can be seen in the confusion matrix that 32 employees were incorrectly predicted as 
    likely to leave and 28 were incorrectly predicted as likely to stay.

    Since decision trees are sensitive to overfitting, the next step would be to test a random forest model (uses an 
    ensemble of decision trees to avoid overfitting).
''')

###

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
rf, y_pred_rf = get_rf(df_dummies, X_train, X_test, y_train)
col1, col2 = st.columns(2)
with col1:
    st_table(get_rf_report())
    st_table(get_rf_metrics())
with col2:
    st_plot(plot_confusion_matrix(y_test, y_pred_rf, rf, 'Random Forest Confusion Matrix'))
st.write('''
    The random forest performed very well despite the imbalance in data. Additionally, training and validation metric 
    scores are very close indicating that the model isn't overfitting and the model predicted significantly less false 
    positives compared to the decision tree.
''')
st_plot(plot_roc(y_test, y_pred_rf))

###

st.subheader('Evaluation')
st_table(get_eval_metrics())
st.write('''
    Precision and F1 scores in the random forest were considerably better than the decision tree and, as seen 
    previously, the random forest had similar false negatives but much less false positives. Thus the random forest is 
    the slightly better model for this application.
''')
st.write('')

st_plot(plot_feature_importance())
st.write('''
    Additionally, it is seen that satisfaction level plays the biggest role in determining whether an employee will
    leave. Last evaluation, number of projects, tenure, and average monthly hours also play a role.
''')

st.divider() #--------------------------------------------------------------------------------------------------------

st.header('Conclusion')
st.write('''
    Many of the issues surrounding employee retention stem from excessive overtime, an overload in projects, and 
    disatisfaction. 
''')
st.subheader('Recommendations')
st.write('''
    - More explicit guidelines surrounding overtime and conversations regarding time off and mental health.
    - Better distribution of projects and communication with employees regarding their ability to complete their
      tasks.
        - Implement a maximum project number of four or five.
    - Review evaluation score metrics, are they fair to all employees or do they favor employees that work more
      projects and hours. Quantity does not always equate to quality.
''')

st.subheader('Next Steps')
st.write('''
    - Consider feature engineering, since the models' scores were very high, data leakage is likely.
    - Evaluate whether gradient boosting would improve performance.
''')

st.divider() #--------------------------------------------------------------------------------------------------------

st.header('Extra')
st.write('''
    Here is small UI to predict whether an employee is likely to leave the company based on the random forest model.
    Feel free to play with the values and see what the model would predict.
''')
predictor.app(rf)

st.divider() #--------------------------------------------------------------------------------------------------------

st.header('Endnote')
linkedin_url = 'https://www.linkedin.com/in/ying-sunwan/'
credential_url = 'https://www.credly.com/badges/92783fb2-9d90-4d7e-9afd-21d5a590aefc/linked_in_profile'
st.write('Thank you for viewing my project! Please check my [LinkedIn](%s) for more' %linkedin_url)
st.write('This project was completed as part of the [Google Advanced Data Analytics Certification](%s) capstone.' %credential_url)
