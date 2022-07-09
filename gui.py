from dataTests import MeasureData
from home import Home
from dataPage import DataPage
from testTable import TestTable
import streamlit as st
import pandas as pd
from io import BytesIO
from streamlit_option_menu import option_menu

st.set_page_config(  # type: ignore[misc]
    page_title="Botpress Testing",
    page_icon=None,
    layout="wide",
    menu_items=None,
)
#####################
#    Definitions    #
#####################



#####################
# Side bar elements #
#####################
training_file = st.sidebar.file_uploader(
    "Training Data (csv)",
    type=["text", "csv"],
    accept_multiple_files=False,
    key="training_data",
    help="The data used to train your model",
    on_change=None,
    args=None,
    kwargs=None,
    disabled=False,
)

testing_file = st.sidebar.file_uploader(
    "Tes File (csv)",
    type=["csv"],
    accept_multiple_files=False,
    key="test_data",
    help="The data used to test your model",
    on_change=None,
    args=None,
    kwargs=None,
    disabled=False,
)


endpoint = st.sidebar.text_input(
    "Endpoint",
    value="http://localhost:3000",  # type: ignore[misc]
    max_chars=None,
    key="endpoint",
    type="default",
    help="The botpress server endpoint",
    autocomplete=None,
    on_change=None,
    args=None,
    kwargs=None,
    placeholder="Place the url of the botpress server here",
    disabled=False,
)

bot_name = st.sidebar.text_input(
    "Bot Name",
    value= "my_super_bot",  # type: ignore[misc]
    max_chars=None,
    key="bot_name",
    type="default",
    help="The Bot Name",
    autocomplete=None,
    on_change=None,
    args=None,
    kwargs=None,
    placeholder="Place the bot name here",
    disabled=False,
)

######################
# Main page elements #
######################
with st.container():
    selected = option_menu(None, ["Home", "Charts", "Table", 'Matrix'], 
        icons=['house', 'bar-chart-fill', "table", 'x-diamond-fill'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    st.title("Meaningful Model Measurements")
try:
   st.session_state['training_df'] = pd.read_csv(training_file).set_index('id')
except (ValueError, AttributeError):
    st.session_state['training_df'] = pd.DataFrame()
try:
    st.session_state['testing_df'] = pd.read_csv(training_file.getValues().decode('utf-8'))
except (ValueError, AttributeError):
    st.session_state['testing_df'] = pd.DataFrame()
## Button to eval data tests

if(selected == 'Home'):
    Home.main()

elif(selected=="Charts"):
    DataPage.main(DataPage)
elif(selected=="Table"):
    TestTable.main(TestTable)