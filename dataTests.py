from typing import Tuple
from xmlrpc.client import boolean
import pandas as pd
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import streamlit as st

# Step 1: get the QNA file and transform it into a table like
# | QNA_ID | Training Phrase |

#data = json.load(open('/Users/gordonclark/Downloads/KeyTexting_RevisedQnas.json'))
@st.cache
class MeasureData:

    def get_csv_from_qnas(file: BytesIO) -> pd.Series:
        """
        Processes an uplaoded qnas.json file from Botpress 

        keyword arguments:
        file - the UploadedFile from the streamlit app

        Returns:
        a Series object with the intent name as index and training phrase as value, like:
        |   id     |    Training Phrase     |
        |----------|------------------------|
        | Hello    | Helllo chatbot!        | 
        """
        qna_df = pd.json_normalize(json.load(file)['qnas'])
        qna_df['id'] = qna_df['id'].apply(lambda x: x[11:])
        qna_df.set_index('id', inplace=True)
        qna_series = qna_df['data.questions.en'].explode()
        
        return qna_series


    def get_distribution(df: pd.DataFrame) -> Tuple[pd.Series, plt.Figure]:
        """ 
        Tests quantity and distribution

        keyword arguments:
        df - the data to test

        returns:
        Tuple(
            agg_df - a pandas Series showing how many phrases are available for each intent
            fig - a histogram with normal curve illustrating the quantity & distribution
        )
        """

        # First get some basic statistics on the dataset
        agg_df = df.index.value_counts()

        agg_df.rename("Count")
        mean = np.mean(agg_df)
        median = np.median(agg_df)
        #mode = pd.Series.value_counts(agg_df).index[0]
        #minimum = agg_df.iloc[-1]
        maximum = agg_df.iloc[0]
        total = agg_df.sum()
        stdev = np.std(agg_df) if np.std(agg_df) != 0 else 1 
        size = agg_df.size
        num_bins = int(np.floor(maximum/stdev))

        

        # Init the histogram
        f = plt.figure(facecolor='white')
        ax = f.add_subplot(111)
        textstr = f"Total: {total} \nMean: {np.round(mean,2)} \nMedian: {median} \nSize: {size} \nSt Dev: {np.round(stdev, 2)}"
        ax.text(0.75, 0.9, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top')
        _,bins,_ = plt.hist(agg_df,num_bins, density=1, alpha=0.5, label="Utterances", figure=f)
        mu, sigma = stats.norm.fit(agg_df)
        best_fit_line = stats.norm.pdf(bins, mu, sigma)*2
        plt.plot(bins, best_fit_line, label='Distribution Curve', figure=f)

        # Add the legend
        legend = plt.legend(title="Distribution of training phrases per intent", 
        ncol=2, loc='upper center', bbox_to_anchor=(0.5,1.18))
        plt.xlabel("Training phrase quantity")
        plt.ylabel("Frequency")

        return (agg_df, f)

    def getTypeDistribution(df: pd.Series) -> dict:
        """ 
        Calculates the length bias and bias type for data. Categories are:
        "Keywords" - phrases 3 words or fewer
        "Natural Language" - phrases between 4 and 12 words long
        "Fluff" - phrases longer than 12 words

        keyword arguments:
        df - the Series of data with the expected intent as index

        Returns:
        a list of length 3 with each value the intent's bias towards keywords, natural language, or fluff
        i.e. [0.15, 0.65, 0.2] means the intent's data s 65% natural language, 20%, fluff, and 15% keywords
        """
        distDict = {}
        for i in df.index.unique().values:
            utterances = np.hstack(df.loc[i].values)
            keyword_count = 0
            natLang_count = 0
            fluff_count = 0
            n = len(utterances)
            for u in utterances:
                wordcount = len(u.split(" "))
                if(wordcount < 4): keyword_count += 1
                elif(5<= wordcount <= 12 ): natLang_count += 1
                elif(wordcount > 12): fluff_count += 1
            distDict[i] = [keyword_count/n, natLang_count/n, fluff_count/n]
        return distDict

    def get_length_bias(self, df: dict) -> go.Figure:
        """ 
        Creates a polar scatterplot that shows the distribution of length bias among an intent's training/testing phrases 
        
        keyword arguments:
        df - a dictionary with intent names as keys and a list of biases as values, like 'hello : [0.65, 0.2, 0.15]'

        returns:
        A plotly polar scatterpot
        """
        demo_data = pd.Series(self.getTypeDistribution(df))
        # Create the figure & scatterplot
        fig3 = go.Figure()
        # Calculate the root sum as the radius
        # The const sqrt(6)/2 comes from normalizing an intent that is 100% fluff or whatever to a value of 1
  
        fig3.add_trace(
            go.Scatterpolar(
                r=demo_data.apply(lambda x:
                    # Calculate the root sum as the radius
                    # The const sqrt(6)/2 comes from normalizing an intent that is 100% fluff or whatever to a value of 1
                    (np.sqrt(np.sum(np.square(np.subtract(x, [1/3, 1/3, 1/3])))) * (np.sqrt(6)/2))
                ),
                theta=demo_data.apply( lambda x:
                    # Calculate theta by multiplying each bias by the appropriate bias axis
                    np.sum(np.multiply(x, [(120*np.pi)/180,(240*np.pi)/180,(360*np.pi)/180]))
                ),
                thetaunit='radians',
                mode='markers',
                hoverinfo='text',
                hovertext = demo_data.index.values,
                name="Training Data"
            )
        )
        # Formatting
        fig3.update_layout(
            # Chart Size
            autosize=False,
            width=900, 
            height=900,
            # Chart Title
            title = 'Distribution of length bias in intents <br><sup>Hover over a datapoint for more informaton</sup>',
            title_font_size=28,
            title_font_color ='black',
            # Background color
            paper_bgcolor='light blue',
            # Legend
            showlegend=True,
            legend=dict(
                font=dict(
                    size=14,
                    color="black"
                ),
                bgcolor="light blue",
            ),
            #Polar Scatter formatting
            polar = dict(
            angularaxis = dict(
                tickmode='array',
                nticks=3,
                tickvals = [0,120,240],
                ticktext = ["Fluff", "Keywords", "Natural Language"],
                tickcolor = 'black',
                color='black'
            )
            )
        )
        # Show the chart
        return fig3

    def get_distribution_table(self, df: pd.Series, min_keyword:float, max_keyword: float,
                                min_natLang:float, max_natLang: float, min_fluff: float, max_fluff:float) -> pd.DataFrame:
        """
        Analyses phrase length distribution per inent and rolls it up into an easy-to-read table with emojis

        keyword arguments:
        df - the raw data in a Series with the intent name as index and phrases as values
        min_keyword - the minimum percentage of keyword (< 4 words ) phrases as a decimal
        max_keyword - the maximum percentage of keyword (< 4 words ) phrases as a decimal
        min_natLang - the minimum percentage of natural language (4 - 12 words) phrases as a decimal
        max_natLang - the maximum percentage of natural language (4 - 12 words) phrases as a decimal
        min_fluff - the minimum percentage of fluff (> 12 words) phrases as a decimal
        max_fluff - the maximum percentage of fluff (> 12 words) phrases as a decimal
        """

        demo_data = pd.DataFrame.from_dict(self.getTypeDistribution(df),orient='index',
                                                columns=['Keywords %', 'Natural Language %', 'Fluff %'])
        demo_data['Keywords %'] = demo_data['Keywords %'].apply(lambda x:  "✅" if (x >= min_keyword and x<= max_keyword) else "🚫")
        demo_data['Natural Language %'] = demo_data["Natural Language %"].apply(lambda x:  "✅" if (x >= min_natLang and x<= max_natLang) else "🚫")
        demo_data['Fluff %'] = demo_data['Fluff %'].apply(lambda x:  "✅" if (x >= min_fluff and x<= max_fluff) else "🚫")
        return demo_data