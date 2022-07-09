import streamlit as st
from dataTests import MeasureData

class DataPage:
    def training_data_tests(self, training_df, testing_file=None):
            """ 
            Main function for running data measurement tests on training data only 

            Keyword Arguments:
            training_df - a pandas dataframe containing model training data
            """
            with st.spinner(text="Processing..."):
                # Table & Histogram w/ Normal Curve
                agg_df, phraseDist = MeasureData.get_distribution(training_df)
                #Polar Scatterplot
                plotlyFig = MeasureData.get_length_bias(MeasureData, training_df)
                return (agg_df, phraseDist, plotlyFig)
        

    def main(self):
        with st.container():
            training_df = st.session_state['training_df']
            testing_df = st.session_state['testing_df']
            if((training_df.size != 0) & (testing_df.size == 0)):
                button1 = st.button(
                        "Show charts for training phrases only",
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
            agg_df, dist, sp = self.training_data_tests(self, training_df)
            col1, col2 = st.columns([2,4])
            with col1:
                st.subheader("Training prases per intent")
                st.dataframe(agg_df)
                
            with col2:
                st.pyplot(dist)
            st.plotly_chart(sp)