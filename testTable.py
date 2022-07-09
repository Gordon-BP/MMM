import streamlit as st
from dataTests import MeasureData

class TestTable:
    def table_tests(self, training_df,min_keyword, max_keyword, min_natLang, max_natLang, min_fluff, max_fluff):
        df = MeasureData.get_distribution_table(MeasureData, training_df, min_keyword, max_keyword, min_natLang, max_natLang, min_fluff, max_fluff)
        return df

    def main(self):
        training_df = st.session_state['training_df']
        with st.container():    
            with st.form(key="lengthDistArgs"):
                st.subheader("Distribution tests per intent")
                # Table of length distribution tests
                col3, col4, col5 = st.columns(3)
                min_keyword = col3.number_input(label="Ideal Minimum Keyword Bias", min_value=0.0, max_value=1.0, value=0.05)
                max_keyword = col3.number_input(label="Ideal Maximum Keyword Bias", min_value=0.0, max_value=1.0, value=0.25)
                min_natLang = col4.number_input(label="Ideal Minimum Natural Language Bias", min_value=0.0, max_value=1.0, value=0.25)
                max_natLang = col4.number_input(label="Ideal Maximum Natural Language Bias", min_value=0.0, max_value=1.0, value=0.65)
                min_fluff = col5.number_input(label="Ideal Minimum Fluff Bias", min_value=0.0, max_value=1.0, value=0.25)
                max_fluff = col5.number_input(label="Ideal Maximum Fluff Bias", min_value=0.0, max_value=1.0, value=0.50)
                
                submit = st.form_submit_button()
                if(submit):
                    st.subheader("Length distribution pass/fail by intent")
                    st.write("Table is interactive- you can click & drag to resize, and click on the header to sort!")
                    df = self.table_tests(self,training_df, min_keyword, max_keyword, min_natLang, max_natLang, min_fluff, max_fluff)
                    st.write(df)