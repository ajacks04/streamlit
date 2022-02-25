import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go

st.set_page_config(page_title ="Forecast Online Case Workload 2022",
                    initial_sidebar_state="collapsed",
                    page_icon="🔮")


tabs = ["Application","About"]
page = st.sidebar.radio("Tabs",tabs)

st.title('Forecast Online Case Workload 2022 🧙🏻')
st.write('This app enables you to generate time series forecast for ICE Online manual workload.')
st.markdown("""The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)**.""")

DATE_COLUMN = 'to_date'
DATA_URL = ('https://raw.githubusercontent.com/ajacks04/streamlit/master/data/all_cases_online.csv')


@st.cache(persist=False,
          allow_output_mutation=True,
          suppress_st_warning=True,
          show_spinner= True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(1000)

st.subheader('1. Data loading 🏋️')      
st.write(data)

data = data.rename(columns={"to_date": "ds", "count_new_cases": "y"})


st.subheader("2. Parameters configuration 🛠️")


with st.container():
  st.write('In this section you can modify the algorithm settings.')
            
  with st.expander("Horizon"):
    periods_input = st.number_input('Select how many future periods (days) to forecast.',
    min_value = 1, max_value = 366,value=90)

  with st.expander("Seasonality"):
    st.markdown("""The default seasonality used is additive, but the best choice depends on the specific case, therefore specific domain knowledge is required. For more informations visit the [documentation](https://facebook.github.io/prophet/docs/multiplicative_seasonality.html)""")
    seasonality = st.radio(label='Seasonality',options=['additive','multiplicative'])

  with st.expander("Trend components"):
    st.write("Add or remove components:")
    weekly= st.checkbox("Weekly")
    monthly = st.checkbox("Monthly")
    yearly = st.checkbox("Yearly")

  with st.expander("Growth model"):
    st.write('Prophet uses by default a linear growth model.')
    st.markdown("""For more information check the [documentation](https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth)""")

    growth = st.radio(label='Growth model',options=['linear',"logistic"]) 

    if growth == 'linear':
      growth_settings= {'cap':1,'floor':0}
      cap=1
      floor=1


    if growth == 'logistic':
      st.info('Configure saturation')

      cap = st.slider('Cap',min_value=0.0,max_value=1.0,step=0.05)
      floor = st.slider('Floor',min_value=0.0,max_value=1.0,step=0.05)
      if floor > cap:
        st.error('Invalid settings. Cap must be higher then floor.')
        growth_settings={}

      if floor == cap:
        st.warning('Cap must be higher than floor')
      else:
        growth_settings = {'cap':cap,'floor':floor}


m = Prophet(seasonality_mode=seasonality)
m.fit(data)
future = m.make_future_dataframe(periods=periods_input)
forecast = m.predict(future)

fig1 = plot_plotly(m, forecast)
fig2 = plot_components_plotly(m, forecast)


st.subheader(f'3. Forecast for {periods_input} days')
st.write(fig1)

st.subheader("4. Forecast components")
st.write(fig2)
