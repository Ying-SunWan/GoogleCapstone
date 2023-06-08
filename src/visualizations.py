import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_tenure_boxplot(df):
    tenure_boxplot = px.box(
        df, 
        y=['tenure'],
        title = 'Tenure Distribution',
        color_discrete_sequence = ['#636EFA'],
    )
    tenure_boxplot.update_layout(
        margin = dict(l=20, r=20, t=40, b=0),
        height = 300,
        xaxis_visible = False,
        yaxis_title = 'Years',
    )
    return tenure_boxplot

def plot_correlations(df):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_mask = corr.mask(mask)

    correlation_heatmap = go.Figure()
    correlation_heatmap.add_trace(
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
    correlation_heatmap.update_layout(
        title = 'Pearson Correlations Between Features',
        margin = dict(l=20, r=20, t=40, b=0),
        height = 400,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed',
    )
    return correlation_heatmap

def plot_hours_per_project(df):
    left = df[df.left==1].groupby('number_project').mean().average_monthly_hours.rename('1')
    stay = df[df.left==0].groupby('number_project').mean().average_monthly_hours.rename('0')
    df1 = pd.concat((stay, left), axis=1, join='outer')
    df1.reset_index(inplace=True)
    hours_per_proj = px.bar(
        df1, 
        x = 'number_project', 
        y = ['0', '1'], 
        barmode = 'group',
        title = 'Average Monthly Hours per Number Projects',
        color_discrete_sequence=['#636EFA', '#FF6692'],
    )
    hours_per_proj.update_layout(
        margin = dict(l=20, r=50, t=30, b=0),
        xaxis_title = 'Number of Projects',
        xaxis_dtick = 1,
        yaxis_title = 'Average Monthly Hours',
        legend_title = 'Left',
        # showlegend = False,
        height = 300,
    )
    return hours_per_proj

def plot_hours_hist(df):
    hours_histogram = px.histogram(
        df, 
        x = 'average_monthly_hours', 
        color = 'left', 
        color_discrete_sequence = ['#FF6692', '#636EFA'],
        histnorm = 'percent',
        nbins = 30,
        title = 'Distribution of Average Monthly Hours',
    )
    hours_histogram.update_layout(
        hovermode='x unified', 
        barmode='overlay',
        height = 300,
        xaxis_title = 'Average Monthly Hours',
        yaxis_title = 'Percent Employees',
        margin = dict(l=20, r=0, t=30, b=0),
        legend = {'traceorder':'reversed'},
        legend_title = 'Left',
    )
    hours_histogram.update_traces(opacity=0.75)
    hours_histogram.add_vline(
        x = df.average_monthly_hours.mean(),
        line_width = 1, 
        line_dash = 'dash', 
        annotation_text = f'total mean = {int(df.average_monthly_hours.median())}',
    )
    return hours_histogram

def plot_satisfaction(df):
    satisfaction_per_hours = px.scatter(
        title = 'Satisfaction Corresponding to Worked Monthly Hours'
    )
    satisfaction_per_hours.add_trace(
        go.Scatter(
            x = df[df.left==0].satisfaction_level, 
            y = df[df.left==0].average_monthly_hours,
            mode = 'markers',
            marker_color = '#636EFA',
            marker_opacity = 0.75,
            name = '0'
        )
    )
    satisfaction_per_hours.add_trace(
        go.Scatter(
            x = df[df.left==1].satisfaction_level, 
            y = df[df.left==1].average_monthly_hours,
            mode = 'markers',
            marker_color = '#FF6692',
            marker_opacity = 0.75,
            name = '1'
        )
    )
    satisfaction_per_hours.update_layout( 
        height = 300,
        margin = dict(l=20, r=0, t=30, b=0),
        xaxis_title = 'Satisfaction Level',
        yaxis_title = 'Average Monthly Hours',
        legend_title = 'Left',
    )
    return satisfaction_per_hours

def plot_tenure_promotions(df):
    left_promo = df[(df.left==1)&(df.promotion_last_5years==1)].groupby('tenure').size().rename('Promoted and Left')
    left_nopromo = df[(df.left==1)&(df.promotion_last_5years==0)].groupby('tenure').size().rename("Wasn't Promoted and Left")
    stay_promo = df[(df.left==0)&(df.promotion_last_5years==1)].groupby('tenure').size().rename('Promoted and Stayed')
    stay_nopromo = df[(df.left==0)&(df.promotion_last_5years==0)].groupby('tenure').size().rename("Wasn't Promoted and Stayed")
    left = pd.concat((left_nopromo, left_promo), axis=1, join='outer')
    stay = pd.concat((stay_nopromo, stay_promo), axis=1, join='outer')
    df1 = pd.concat((stay, left), axis=1, join='outer')

    tenure_promo = px.bar(
        df1,
        color_discrete_sequence = ['#FF6692', '#FFCCEA', '#AB63FA', '#636EFA'],
        title = 'Promotions by Tenure',
    )
    tenure_promo.update_layout( 
        hovermode = 'x unified',
        height = 300,
        margin = dict(l=20, r=0, t=30, b=0),
        xaxis_title = 'Tenure',
        yaxis_title = 'Number of Employees',
        legend_title = '',
    )
    return tenure_promo

def plot_num_projects(df):
    left = df[df.left==1].groupby('number_project').size().rename('1')
    stay = df[df.left==0].groupby('number_project').size().rename('0')
    df1 = pd.concat((stay, left), axis=1, join='outer')
    df1.reset_index(inplace=True)

    num_proj = px.bar(
        df1, 
        x = 'number_project', 
        y = ['0', '1'], 
        barmode = 'group',
        title = 'Number of Projects Assigned to Employees',
        color_discrete_sequence = ['#636EFA', '#FF6692'],
    )
    num_proj.update_layout(
        margin = dict(l=20, r=0, t=30, b=0),
        xaxis_title = 'Number of Projects',
        xaxis_dtick = 1,
        yaxis_title = 'Number of Employees',
        legend_title = 'Left',
        height = 300
    )
    return num_proj

def plot_eval_per_project(df):
    show2 = df[df.number_project==2]
    show6 = df[df.number_project==6]
    hide = df[(df.number_project!=2)&(df.number_project!=6)].sort_values('number_project')
    eval_per_project = px.histogram(
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
        eval_per_project.add_trace(trace)
    show6 = px.histogram(
        show6, 
        x = 'last_evaluation', 
        color = 'number_project', 
        color_discrete_sequence = ['#FF6692'],
        histnorm = 'percent', 
        nbins=20, 
    )
    eval_per_project.add_trace(show6.data[0])
    eval_per_project.add_trace(hide.data[-1])
    eval_per_project.update_layout(
        hovermode = 'x unified', 
        barmode = 'overlay', 
        legend = {'traceorder':'normal'},
        margin = dict(l=20, r=0, t=30, b=0),
        xaxis_title = 'Last Evaluation',
        yaxis_title = 'Percent Employees'
    )
    eval_per_project.update_traces(opacity=0.75)
    return eval_per_project

def plot_confusion_matrix(y_test, y_pred, model, title):
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cm_plot = go.Figure()
    cm_plot.add_trace(
        go.Heatmap(
            z = cm, 
            x = ['0', '1'], 
            y = ['0', '1'], 
            texttemplate = '%{z:f}',
            autocolorscale = False,
            colorscale = ['#636EFA', '#FFCCEA'],
        ),
    )
    cm_plot.update_layout(
        height = 310,
        title = title,
        margin = dict(l=0, r=10, t=30, b=0),
    )
    cm_plot.update_xaxes(
        title = 'Predicted',
        type = 'category',
    )
    cm_plot.update_yaxes(
        title = 'True',
        type = 'category',
    )
    return cm_plot

def plot_roc(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc = px.area(
        x = fpr, 
        y = tpr,
        title = f'Random Forest ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels = dict(
            x = 'False Positive Rate', 
            y = 'True Positive Rate',
        ),
        color_discrete_sequence = ['#636EFA'],
    )
    roc.add_shape(
        type = 'line', 
        line = dict(
            dash = 'dash',
            color = '#636EFA',
        ),
        x0 = 0, x1 = 1, 
        y0 = 0, y1 = 1,
    )
    roc.update_layout(
        margin = dict(l=10, r=10, t=30, b=0),
        height = 300,
    )
    return roc

def plot_feature_importance():
    importance = pd.DataFrame({
        'Feature': [
            'satisfaction_level', 
            'last_evaluation', 
            'number_project', 
            'tenure', 
            'average_monthly_hours', 
            'salary_low', 
            'department_sales', 
            'department_technical', 
            'salary_medium', 
            'department_support',
        ],
        'Importance': [
            0.4711034,
            0.153502,
            0.150336,
            0.1168844,
            0.09440478,
            0.002121856,
            0.001981863,
            0.001867283,
            0.001731531,
            0.001179128,
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
    return feature_importance