import pandas as pd
import os
import operator
from build_model import MovieGroupProcess
import gensim
import numpy as np
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
np.random.seed(2018)
import pickle
stemmer = PorterStemmer()
from spellchecker import SpellChecker
spell = SpellChecker()
STOPWORDS = list(STOPWORDS)
STOPWORDS.append('covid')
STOPWORDS.append('coronavirus')
STOPWORDS.append('corona')
STOPWORDS.append('uganda')
from utils import preprocess, BreakIt, produce_mapping
from apiclient import discovery
from google.oauth2 import service_account
from datetime import datetime
from time import sleep


def main():
    # initialize google sheets API
    scopes = ['https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_file('service_account_key.json', scopes=scopes)
    service = discovery.build('sheets', 'v4', credentials=credentials)

    # load data
    spreadsheetId = '18PwsExSVerYzTxGxarLwyGkKIVT2QJCobCnoeLYXwjM'
    rangeName = 'input!A:F'
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheetId, range=rangeName).execute()
    values = result.get('values', [])
    df = pd.DataFrame.from_records(values)[1:] # convert to pandas dataframe
    df[4:5] = df[4:5].astype(str)
    df = df.replace('None', '')
    df = df.replace(np.nan, '')
    df["rumors"] = df[4] + df[5]
    df_input = df.copy()

    df = df[df.rumors != 'None']
    df = df[df.rumors != 'None ']
    df = df[df.rumors != 'No rumors']
    df = df[df.rumors.str.len() > 4]

    text = df["rumors"]
    len_original = len(text)
    text_split = pd.DataFrame()
    for index, row in df.iterrows():
        split_text = row['rumors'].split('\n')
        if len(split_text) > 0:
            text_split = text_split.append(pd.Series({'text': split_text[0],
                                                      'survey_id': index}), ignore_index=True)
            for more in split_text[1:]:
                text_split = text_split.append(pd.Series({'text': more,
                                                          'survey_id': index}), ignore_index=True)
    text_split = text_split.reset_index(drop=True)
    text = text_split['text']

    # pre-process text
    processed_ser = text.map(preprocess)
    processed_docs = [item[0] for item in processed_ser]
    mapping_list = [item[1] for item in processed_ser]
    mapping121, mapping12many = produce_mapping(mapping_list)

    # load GSDMM model
    model = pickle.load(open("gsdmm_model.pickle", "rb"))

    # create list of topic descriptions (lists of keywords) and scores
    matched_topic_score_list = [model.choose_best_label(i) for i in processed_docs]
    matched_topic_list = [t[0] for t in matched_topic_score_list]
    score_list = [t[1] for t in matched_topic_score_list]
    text = pd.DataFrame({'text': text.values, 'topic_num': matched_topic_list, 'score': score_list})

    # create list of human-readable topic descriptions (de-lemmatize)
    topic_list = [list(reversed(sorted(x.items(), key=operator.itemgetter(1))[-5:])) for x in model.cluster_word_distribution]
    topic_list_flat = [[l[0] for l in x] for x in topic_list]

    # create dataframe with best example per topic and topic description
    df = pd.DataFrame()
    for topic_num, topic in enumerate(topic_list_flat):
        text_topic = text[text.topic_num == topic_num].sort_values(by=['score'], ascending=False).reset_index()
        frequency = len(text[text.topic_num == topic_num]) / len_original
        responses = len(text[text.topic_num == topic_num])
        if not text_topic.empty:
            df = df.append(pd.Series({"topic number": int(topic_num),
                                      "frequency (%)": int(frequency * 100.),
                                      "number of responses": responses}), ignore_index=True)
    df = df.sort_values(by=['frequency (%)'], ascending=False)

    # update data

    # add topic number to individual entries
    df_input['topic_number'] = ''
    for ix, row in text_split.iterrows():
        survey_id = int(row['survey_id'])
        df_input.at[survey_id, 'topic_number'] = matched_topic_list[ix]
    df_input['topic_number'] = df_input['topic_number'].astype(str)

    for i in range(0, round(len(df_input)/100)):
        start = i*100
        end = 100+i*100
        data_to_upload = df_input[start:end][['topic_number']].values.tolist()
        TargetRangeName = f'input!G{start+2}:G{end+2}'
        body = {
            "range": TargetRangeName,
            "values": data_to_upload
        }
        value_input_option = 'USER_ENTERED'
        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheetId, range=TargetRangeName, valueInputOption=value_input_option, body=body).execute()
        sleep(2)

    # update the output sheet with aggregated numbers
    rangeName = 'output!D:F'
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheetId, range=rangeName).execute()
    values = result.get('values', [])
    df_results = pd.DataFrame.from_records(values)[1:] # convert to pandas dataframe
    df_results = df_results.rename(columns={0: "frequency (%)", 1: "number of responses", 2: "topic number"})
    df_results["topic number"] = df_results["topic number"].astype(int)

    for ix, row in df_results.iterrows():
        df_results.at[ix, "frequency (%)"] = df[df["topic number"] == row["topic number"]]["frequency (%)"].values[0]
        df_results.at[ix, "number of responses"] = df[df["topic number"] == row["topic number"]]["number of responses"].values[0]

    # reformat data and push to google sheets
    data_to_upload = [['frequency (%)', 'number of responses']] + \
                     df_results[['frequency (%)', 'number of responses']].values.tolist()

    TargetRangeName = 'output!D:E'
    body = {
        "range": TargetRangeName,
        "values": data_to_upload
    }
    value_input_option = 'USER_ENTERED'
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheetId, range=TargetRangeName, valueInputOption=value_input_option, body=body).execute()

    # add metadata
    metadata = [[datetime.now().strftime("%m/%d/%Y, %H:%M:%S")]]
    TargetRangeName = 'metadata!A:A'
    body = {
        "range": TargetRangeName,
        "values": metadata
    }
    value_input_option = 'USER_ENTERED'
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheetId, range=TargetRangeName, valueInputOption=value_input_option, body=body).execute()

    print("Forecast update was a Success!", datetime.now())


if __name__ == "__main__":
    n = 5
    while n > 0:
        main()
        sleep(86400)

