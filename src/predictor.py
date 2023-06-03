import pandas as pd
import streamlit as st

def app(rf):
    '''
    Streamlit UI for predicting whether a employee will leave or not based on the random forest model.
    '''
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