import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pydeck as pdk
import seaborn as sns
from prophet import Prophet
import numpy as np


# Load the cleaned data
@st.cache_data
def load_data():
    return pd.read_excel("Dorset Police Data_cleaned.xlsx")

df_clean = load_data()
df_clean['Location'] = df_clean['Location'].str.replace('On or near', '').str.strip()
df_clean['Location'] = df_clean['Location'].fillna('No Location')


# Display Dorset Police logo in the sidebar
st.sidebar.image("https://www.dorset.police.uk/SysSiteAssets/media/images/brand/dorset/crest/dorset-police-logo-blue-sitelogo-217px.png", use_container_width=True)

# Sidebar Header
st.sidebar.header("Filters")

# Year filter with "All" option
years = sorted(df_clean['Year'].dropna().unique())
years.insert(0, 'All')
selected_year = st.sidebar.selectbox("Select Year", years)

# Month filter with "All"
months = sorted(df_clean['Month'].dropna().unique())
months.insert(0, 'All')
selected_month = st.sidebar.selectbox("Select Month", months)

# Crime type filter with "All"
crime_types = sorted(df_clean['Crime type'].dropna().unique())
crime_types.insert(0, 'All')
selected_crime = st.sidebar.selectbox("Select Crime Type", crime_types)

# Location filter with "All"

df_clean['Location'] = df_clean['Location'].replace(r'^\s*$', np.nan, regex=True)
locations = sorted(df_clean['Location'].dropna().unique())
locations.insert(0, 'All')
selected_location = st.sidebar.selectbox("Select Location", locations)

# Filter data based on selections
filtered_df = df_clean.copy()

if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['Year'] == selected_year]
if selected_month != 'All':
    filtered_df = filtered_df[filtered_df['Month'] == selected_month]
if selected_crime != 'All':
    filtered_df = filtered_df[filtered_df['Crime type'] == selected_crime]
if selected_location != 'All':
    filtered_df = filtered_df[filtered_df['Location'] == selected_location]



#Summary Metrics Section
st.markdown("""
<style>
.summary-container {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    margin-bottom: 20px;
}
.summary-card {
    background-color: #f5f5f5;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    text-align: center;
    flex: 1;
    min-height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.summary-title {
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 10px;
}
.summary-value {
    font-size: 18px;
    color: #4a90e2;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2; /* limit to 2 lines */
    -webkit-box-orient: vertical;
    line-height: 1.2em;
    max-height: 2.4em;
}
</style>
""", unsafe_allow_html=True)


# --- Horizontal Tabs ---
tabs = st.tabs(["Home", "Crime Distribution", "Trends & Top Crimes", "Interactive Map", "Forecast"])

# --- Tab 0: Home / Welcome ---
with tabs[0]:
    st.markdown("""
   <div style="text-align: center; margin-bottom: 20px;">
       <img src="https://www.dorset.police.uk/SysSiteAssets/media/images/brand/dorset/crest/dorset-police-logo-blue-sitelogo-217px.png" 
            width="300">
   </div>
   """, unsafe_allow_html=True)
    st.subheader("Welcome to the Dorset Police Data Dashboard")
    st.markdown("_A high-level overview of reported crimes in Dorset._")

    # Summary cards
    st.markdown('<div class="summary-container">', unsafe_allow_html=True)

    total_crimes = df_clean.shape[0]
    most_common_crime = df_clean['Crime type'].mode()[0] if not df_clean['Crime type'].mode().empty else "N/A"
    most_common_outcome = df_clean['Last outcome category'].mode()[0] if not df_clean['Last outcome category'].mode().empty else "N/A"
    unique_locations = df_clean['Location'].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-title">Total Crimes</div>
            <div class="summary-value">{total_crimes:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-title">Most Common Crime</div>
            <div class="summary-value">{most_common_crime}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-title">Most Common Outcome</div>
            <div class="summary-value">{most_common_outcome}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-title">Unique Crime Locations</div>
            <div class="summary-value">{unique_locations:,}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Tab 1: Crime Distribution ---
with tabs[1]:
    st.subheader("Crime Distribution by Location and Outcome")


    st.subheader("Crime Hotspot")
    st.markdown("The most common locations of criminal incidents by crime counts.")

    loc_df = filtered_df.copy()
    street_counts = loc_df['Location'].value_counts().head(10).reset_index()
    street_counts.columns = ['Location', 'Count']

    fig1 = px.bar(
        street_counts,
        x='Location',
        y='Count',
        title="Top 10 Streets by Crime Count",
        color_discrete_sequence=px.colors.sequential.Viridis,
        height=700,
        )
    fig1.update_layout(xaxis_tickangle=-90) # Rotates x-axis labels
    st.plotly_chart(fig1)

  
    st.subheader("Outcome Category")
    st.markdown("A categorical breakdown of crime resolutions")
    outcome_counts = filtered_df['Last outcome category'].value_counts().reset_index()
    outcome_counts.columns = ['Outcome Category', 'Count']    
    fig2 = px.bar(
        outcome_counts,
        x='Outcome Category',
        y='Count',
        title="Crime Outcomes by Category",
        color_discrete_sequence=['lightcoral'],
        height=700,
        )   

# Rotate x-axis labels for readability if needed
    fig2.update_layout(xaxis_tickangle=-90)

# Display the chart in your Streamlit app
    st.plotly_chart(fig2)

# --- Tab 2: Trends & Top Crimes ---
with tabs[2]:
    st.subheader("Trends and Top Crimes")
    col1, col2 = st.columns([1,1])

    # Crimes over time
    with col1:
        st.subheader("Crimes Over Time")
        st.markdown("_The trend of reported crimes across months._")
        date_order = sorted(df_clean['Date'].dropna().unique(), key=lambda x: pd.to_datetime(x, format='%B %Y'))
        monthly_crimes = df_clean['Date'].value_counts().reindex(date_order, fill_value=0).reset_index()
        monthly_crimes.columns = ['Date','Count']

        fig_line = px.line(monthly_crimes, x="Date", y="Count", markers=True, line_shape="linear")
        fig_line.update_layout(height=400, margin=dict(t=30,b=30,l=10,r=10),
                               xaxis_title="Month-Year", yaxis_title="Number of Crimes")
        fig_line.update_traces(line=dict(color="darkorange", width=2))
        st.plotly_chart(fig_line, use_container_width=True)

    # Top 5 crime types
    with col2:
        st.subheader("Top 5 Crime Types")
        st.markdown("_The five most common crime types make up the majority of incidents._")
        top_crimes = filtered_df['Crime type'].value_counts().nlargest(5).reset_index()
        top_crimes.columns = ['Crime Type','Count']

        fig_bar = px.bar(top_crimes, x='Crime Type', y='Count', text='Count', color='Crime Type', color_discrete_sequence=px.colors.sequential.RdBu)
        fig_bar.update_layout(height=400, showlegend=False, margin=dict(t=30,b=30,l=10,r=10),
                              xaxis_title="Crime Type", yaxis_title="Number of Crimes")
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

# --- Tab 3: Interactive Map ---
with tabs[3]:
    st.subheader("Interactive Crime Map")
    st.markdown("_Each red dot shows the location of a reported crime for the selected filters._")

    map_df = filtered_df.dropna(subset=['Latitude','Longitude'])
    fig_map = px.scatter_mapbox(
            map_df,
            lat="Latitude",
            lon="Longitude",
            color="Crime type",
            hover_data=["LSOA name", "Last outcome category"],
            zoom=10,
            height=650,
            mapbox_style="carto-positron"
        )
    st.plotly_chart(fig_map, use_container_width=True)

    
    
# --- Tab 4: Crime Forecast ---
with tabs[4]:
    st.subheader("Crime Forecast")
    st.markdown("_Forecasting future crimes based on historical trends._")

    # Prepare data for forecasting
    df_forecast = df_clean[['Date']].copy()
    df_forecast['Count'] = 1
    df_forecast = df_forecast.groupby('Date').count().reset_index()
    df_forecast.columns = ['ds', 'y']  # Prophet requires 'ds' for date, 'y' for value

    # Fit Prophet model
    from prophet import Prophet
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_forecast)

    # Make future dataframe for 12 months
    future = m.make_future_dataframe(periods=12, freq='M')
    forecast = m.predict(future)

    # Plot forecast
    from prophet.plot import plot_plotly
    st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
    

    
    
