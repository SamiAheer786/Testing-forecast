import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go

def calculate_target_analysis(df, target):
    current_sales = df[df['Date'] <= pd.Timestamp.today()]['Sales'].sum()
    forecasted_sales = df[df['Date'] > pd.Timestamp.today()]['Sales'].sum()
    return {
        "Target": "{:,}".format(round(target)),
        "Current Sales": "{:,}".format(round(current_sales)),
        "Forecasted Sales": "{:,}".format(round(forecasted_sales))
    }

def generate_daily_table(df):
    future_df = df[df['Date'] > pd.Timestamp.today()].copy()
    future_df["Forecasted Sales"] = future_df["Sales"].apply(lambda x: "{:,}".format(int(round(x))))
    return future_df[["Date", "Forecasted Sales"]].reset_index(drop=True)

def forecast_by_region(df):
    future_df = df[df['Date'] > pd.Timestamp.today()].copy()
    grouped = future_df.groupby("Region")["Sales"].sum().reset_index()
    grouped.rename(columns={"Sales": "Forecasted Sales"}, inplace=True)
    grouped["Forecasted Sales"] = grouped["Forecasted Sales"].apply(lambda x: "{:,}".format(int(round(x))))
    return grouped

def generate_line_chart(df):
    df_copy = df.copy()
    df_copy['Sales'] = df_copy['Sales'].apply(lambda x: round(x))
    fig = px.line(df_copy, x="Date", y="Sales", title="Sales Forecast", color_discrete_sequence=['#00A8E8'])
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        title_x=0.5,
        template="plotly_white"
    )
    fig.update_yaxes(tickformat=",d")
    return fig

def generate_region_chart(df):
    df_future = df[df['Date'] > pd.Timestamp.today()].copy()
    df_future['Sales'] = df_future['Sales'].apply(lambda x: round(x))
    df_grouped = df_future.groupby("Region")["Sales"].sum().reset_index()
    fig = px.bar(df_grouped, x="Region", y="Sales", title="Forecasted Sales by Region", color_discrete_sequence=['#0077B6'])
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Forecasted Sales",
        title_x=0.5,
        template="plotly_white"
    )
    fig.update_yaxes(tickformat=",d")
    return fig
