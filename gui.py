from nlu_stats import MeasureData
import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(  # type: ignore[misc]
    page_title="Botpress Testing",
    page_icon=None,
    layout="wide",
    menu_items=None,
)
#####################
#    Definitions    #
#####################
def data_tests(training_df, testing_file=None):
    """ Main function for running data measurement tests """
    with st.spinner(text="Processing..."):
        # Table & Histogram w/ Normal Curve
        agg_df, phraseDist = MeasureData.get_distribution(training_df)
        #Polar Scatterplot
        plotlyFig = MeasureData.get_length_bias(MeasureData, training_df)
        return (agg_df, phraseDist, plotlyFig)
       

def table_tests(training_df,min_keyword, max_keyword, min_natLang, max_natLang, min_fluff, max_fluff):
    df = MeasureData.get_distribution_table(MeasureData, training_df, min_keyword, max_keyword, min_natLang, max_natLang, min_fluff, max_fluff)
    return df

def clean_qna_json(jsonFile):
    """ Takes an exported QNA.json file from Botpress and converts it into a two-column CSV needed for processing"""
    st.spinner(text="Processing...")
    print(type(training_file))
    df1 = training_file
    df1['id'] = df1['id'].apply( lambda x : x[11:])
    df1.set_index('id', inplace=True)
    df1 = df1.explode('data.questions.en')
    df2 = df1['data.questions.en']
    st.write(df2)
    st.success("QNA.json file cleaned & ready!")
    return df2

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
    st.title("NLU Stats")
try:
    training_df = pd.read_csv(training_file)
except (ValueError, AttributeError):
    training_df = pd.DataFrame()
try:
    testing_df = pd.read_csv(training_file.getValues().decode('utf-8'))
except (ValueError, AttributeError):
    testing_df = pd.DataFrame()
## Button to eval data tests
with st.container():
    if((training_df.size != 0) & (testing_df.size == 0)):
       button1 = st.button(
            "Show chars for training phrases only",
            key="training_only",
            help="Run tests on training data only",
            kwargs=None,
            disabled=False,
        )
    elif((training_df.size == 0) & (testing_df.size != 0)):
        button1 = st.button(
            "Show charts for testing phrases only",
            key="testing_only",
            help="Run tests on testing data only",
            on_click=st.write,
            args=testing_df,
        )
    elif((training_df.size != 0) & (testing_df.size != 0)):
        button1 = st.button(
            "Show charts for both training and testing phrases",
            key="measure_both",
            help="Run tests to compare training and testing data",
            on_click=st.write,
            args=(training_df, testing_df),
            kwargs=None,
            disabled=False,
        )
    else:
        button1 = None
        st.write("Upload a CSV of your training or testing data (or both!)")

if(button1):
    agg_df, dist, sp = data_tests(training_df)
    col1, col2 = st.columns([2,4])
    with col1:
        st.subheader("Training prases per intent")
        st.dataframe(agg_df)
        
    with col2:
        st.pyplot(dist)
    st.plotly_chart(sp)
with st.container():
    st.subheader("Distribution tests per intent")
    # Table of length distribution tests
    col3, col4, col5 = st.columns(3)
    min_keyword = col3.number_input(label="Ideal Minimum Keyword Bias", min_value=0.0, max_value=1.0, value=0.05)
    max_keyword = col3.number_input(label="Ideal Maximum Keyword Bias", min_value=0.0, max_value=1.0, value=0.25)
    min_natLang = col4.number_input(label="Ideal Minimum Natural Language Bias", min_value=0.0, max_value=1.0, value=0.25)
    max_natLang = col4.number_input(label="Ideal Maximum Natural Language Bias", min_value=0.0, max_value=1.0, value=0.65)
    min_fluff = col5.number_input(label="Ideal Minimum Fluff Bias", min_value=0.0, max_value=1.0, value=0.25)
    max_fluff = col5.number_input(label="Ideal Maximum Fluff Bias", min_value=0.0, max_value=1.0, value=0.50)
    if((training_df.size != 0) & (testing_df.size == 0)):
        button2 = st.button(
            "Test training phrases only",
            key="table_training_only",
            help="Run tests on training data only",
        )
    elif((training_df.size == 0) & (testing_df.size != 0)):
        print("flag2")
        button2 = st.button(
            "Test testing phrases only",
            key="table_testing_only",
            help="Run tests on testing data only",
        )
    elif((training_df.size != 0) & (testing_df.size != 0)):
        print("flag3")
        button2 = st.button2(
            "Test both training and testing phrases",
            key="table_measure_both",
            help="Run tests to compare training and testing data",
        )
    else:
        button2 = None
    if(button2):
        st.subheader("Length distribution pass/fail by intent")
        st.write("Table is interactive- you can click & drag to resize, and click on the header to sort!")
        df = table_tests(training_df, min_keyword, max_keyword, min_natLang, max_natLang, min_fluff, max_fluff)
        st.write(df)