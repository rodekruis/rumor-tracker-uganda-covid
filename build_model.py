import pandas as pd
import os
import operator
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
from numpy.random import multinomial
from numpy import log, exp
from numpy import argmax


class MovieGroupProcess:
    def __init__(self, K=8, alpha=0.1, beta=0.1, n_iters=30):
        '''
        A MovieGroupProcess is a conceptual model introduced by Yin and Wang 2014 to
        describe their Gibbs sampling algorithm for a Dirichlet Mixture Model for the
        clustering short text documents.
        Reference: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf
        Imagine a professor is leading a film class. At the start of the class, the students
        are randomly assigned to K tables. Before class begins, the students make lists of
        their favorite films. The teacher reads the role n_iters times. When
        a student is called, the student must select a new table satisfying either:
            1) The new table has more students than the current table.
        OR
            2) The new table has students with similar lists of favorite movies.
        :param K: int
            Upper bound on the number of possible clusters. Typically many fewer
        :param alpha: float between 0 and 1
            Alpha controls the probability that a student will join a table that is currently empty
            When alpha is 0, no one will join an empty table.
        :param beta: float between 0 and 1
            Beta controls the student's affinity for other students with similar interests. A low beta means
            that students desire to sit with students of similar interests. A high beta means they are less
            concerned with affinity and are more influenced by the popularity of a table
        :param n_iters:
        '''
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters

        # slots for computed variables
        self.number_docs = None
        self.vocab_size = None
        self.cluster_doc_count = [0 for _ in range(K)]
        self.cluster_word_count = [0 for _ in range(K)]
        self.cluster_word_distribution = [{} for i in range(K)]

    @staticmethod
    def from_data(K, alpha, beta, D, vocab_size, cluster_doc_count, cluster_word_count, cluster_word_distribution):
        '''
        Reconstitute a MovieGroupProcess from previously fit data
        :param K:
        :param alpha:
        :param beta:
        :param D:
        :param vocab_size:
        :param cluster_doc_count:
        :param cluster_word_count:
        :param cluster_word_distribution:
        :return:
        '''
        mgp = MovieGroupProcess(K, alpha, beta, n_iters=30)
        mgp.number_docs = D
        mgp.vocab_size = vocab_size
        mgp.cluster_doc_count = cluster_doc_count
        mgp.cluster_word_count = cluster_word_count
        mgp.cluster_word_distribution = cluster_word_distribution
        return mgp

    @staticmethod
    def _sample(p):
        '''
        Sample with probability vector p from a multinomial distribution
        :param p: list
            List of probabilities representing probability vector for the multinomial distribution
        :return: int
            index of randomly selected output
        '''
        return [i for i, entry in enumerate(multinomial(1, p)) if entry != 0][0]

    def fit(self, docs, vocab_size):
        '''
        Cluster the input documents
        :param docs: list of list
            list of lists containing the unique token set of each document
        :param V: total vocabulary size for each document
        :return: list of length len(doc)
            cluster label for each document
        '''
        alpha, beta, K, n_iters, V = self.alpha, self.beta, self.K, self.n_iters, vocab_size

        D = len(docs)
        self.number_docs = D
        self.vocab_size = vocab_size

        # unpack to easy var names
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution
        cluster_count = K
        d_z = [None for i in range(len(docs))]
        total_transfers_old, count_convergence = 1e24, 0

        # initialize the clusters
        for i, doc in enumerate(docs):

            # choose a random  initial cluster for the doc
            z = self._sample([1.0 / K for _ in range(K)])
            d_z[i] = z
            m_z[z] += 1
            n_z[z] += len(doc)

            for word in doc:
                #print(word)
                if word not in n_z_w[z]:
                    n_z_w[z][word] = 0
                n_z_w[z][word] += 1
        #print(n_z_w)
        for _iter in range(n_iters):
            total_transfers = 0

            for i, doc in enumerate(docs):

                # remove the doc from it's current cluster
                z_old = d_z[i]

                m_z[z_old] -= 1
                n_z[z_old] -= len(doc)

                for word in doc:
                    n_z_w[z_old][word] -= 1

                    # compact dictionary to save space
                    if n_z_w[z_old][word] == 0:
                        del n_z_w[z_old][word]

                # draw sample from distribution to find new cluster
                p = self.score(doc)
                z_new = self._sample(p)

                # transfer doc to the new cluster
                if z_new != z_old:
                    total_transfers += 1

                d_z[i] = z_new
                m_z[z_new] += 1
                n_z[z_new] += len(doc)

                for word in doc:
                    if word not in n_z_w[z_new]:
                        n_z_w[z_new][word] = 0
                    n_z_w[z_new][word] += 1

            cluster_count_new = sum([1 for v in m_z if v > 0])
            # print("In stage %d: transferred %d clusters with %d clusters populated" % (
            # _iter, total_transfers, cluster_count_new))

            if abs(total_transfers - total_transfers_old) < 0.1 * 0.5 * (total_transfers+total_transfers_old):
                count_convergence += 1
            else:
                count_convergence = 0
            if count_convergence > 50:
                print("Converged.  Breaking out.")
                break
            total_transfers_old = total_transfers
            # if total_transfers == 0 and cluster_count_new == cluster_count and _iter > 25:
            #     print("Converged.  Breaking out.")
            #     break

            self.cluster_count = cluster_count_new
        self.cluster_word_distribution = n_z_w
        return d_z

    def score(self, doc):
        '''
        Score a document
        Implements formula (3) of Yin and Wang 2014.
        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf
        :param doc: list[str]: The doc token stream
        :return: list[float]: A length K probability vector where each component represents
                              the probability of the document appearing in a particular cluster
        '''
        alpha, beta, K, V, D = self.alpha, self.beta, self.K, self.vocab_size, self.number_docs
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution

        p = [0 for _ in range(K)]

        #  We break the formula into the following pieces
        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        #  lN1 = log(m_z[z] + alpha)
        #  lN2 = log(D - 1 + K*alpha)
        #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))
        #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))

        lD1 = log(D - 1 + K * alpha)
        doc_size = len(doc)
        for label in range(K):
            lN1 = log(m_z[label] + alpha)
            lN2 = 0
            lD2 = 0
            for word in doc:
                lN2 += log(n_z_w[label].get(word, 0) + beta)
            for j in range(1, doc_size +1):
                lD2 += log(n_z[label] + V * beta + j - 1)
            p[label] = exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        pnorm = sum(p)
        pnorm = pnorm if pnorm>0 else 1
        return [pp/pnorm for pp in p]

    def choose_best_label(self, doc):
        '''
        Choose the highest probability label for the input document
        :param doc: list[str]: The doc token stream
        :return:
        '''
        p = self.score(doc)
        return argmax(p), max(p)


def keywords_to_themes(df):
    """assign a description ('theme') to each topic based on keywords"""
    for ix, row in df.iterrows():
        keys = row['keywords']
        if 'kill' in keys and ('whites' in keys or 'people' in keys):
            theme = "it is not dangerous"
        elif ('political' in keys or 'politics' in keys) and ('exist' in keys or 'real' in keys):
            theme = "it does not exist, it is made up by politicians"
        elif 'alcohol' in keys or 'water' in keys or 'drinking' in keys or 'eating' in keys:
            theme = "Alternative cures"
        elif 'manufactured' in keys or 'china' in keys or 'chinese' in keys:
            theme = "Origin of the disease"
        elif 'government' in keys or 'money' in keys or 'business' in keys:
            theme = "Government and/or others are profiting from it"
        elif 'masks' in keys or 'face' in keys or 'god' in keys:
            theme = "Misunderstanding of PPE and/or preventive measures"
        elif 'flu' in keys and 'like' in keys:
            theme = "it is not dangerous"
        else:
            theme = "Unclear / unknown"
        print(keys, '-->', theme)
        df.at[ix, "theme"] = theme
    return df


def main():
    # initialize google sheets API
    scopes = ['https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_file('service_account_key.json', scopes=scopes)
    service = discovery.build('sheets', 'v4', credentials=credentials)

    # load data
    # # df = pd.read_excel('data.xlsx', sheet_name='Volunteer_Tool_v5_September_21')
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

    text = df["rumors"]
    len_original = len(text)
    text = text[text != 'None'].astype(str)
    text = text.tolist()
    for index, value in enumerate(text):
        split_text = value.split('\n')
        if len(split_text) > 1:
            text[index] = split_text[0]
            for more in split_text[1:]:
                text.extend(more)
    text = pd.Series(text)
    text = text[text.str.len() > 4]

    if os.path.exists('processed_text.txt'):
        with open("processed_text.txt", "rb") as fp:  # Unpickling
            processed_ser = pickle.load(fp)
    else:
        processed_ser = text.map(preprocess)
        with open("processed_text.txt", "wb") as fp:  # Pickling
            pickle.dump(processed_ser, fp)
    processed_docs = [item[0] for item in processed_ser]
    mapping_list = [item[1] for item in processed_ser]
    mapping121, mapping12many = produce_mapping(mapping_list)

    # initialize and fit GSDMM model
    if os.path.exists('gsdmm_model.pickle'):
        model = pickle.load(open("gsdmm_model.pickle", "rb"))
    else:
        print('initialize and fit GSDMM model')
        model = MovieGroupProcess(K=10, alpha=0.3, beta=0.05, n_iters=500)
        y = model.fit(processed_docs, len(processed_docs))
        pickle.dump(model, open("gsdmm_model.pickle", "wb"))

    # create list of topic descriptions (lists of keywords) and scores
    matched_topic_score_list = [model.choose_best_label(i) for i in processed_docs]
    matched_topic_list = [t[0] for t in matched_topic_score_list]
    score_list = [t[1] for t in matched_topic_score_list]
    text = pd.DataFrame({'text': text.values, 'topic_num': matched_topic_list, 'score': score_list})

    # create list of human-readable topic descriptions (de-lemmatize)
    print('create list of human-readable topic descriptions (de-lemmatize)')
    topic_list = [list(reversed(sorted(x.items(), key=operator.itemgetter(1))[-5:])) for x in model.cluster_word_distribution]
    topic_list_flat = [[l[0] for l in x] for x in topic_list]
    topic_list_human_readable = topic_list_flat.copy()
    for ixt, topic in enumerate(topic_list_human_readable):
        for ixw, word in enumerate(topic):
            try:
                for raw in text.text.values:
                    for token in gensim.utils.simple_preprocess(raw):
                        if word in token:
                            topic_list_human_readable[ixt][ixw] = token
                            raise BreakIt
            except BreakIt:
                pass
    topic_list_human_readable = [[spell.correction(t) for t in l] for l in topic_list_human_readable]

    # create dataframe with best example per topic and topic description
    print('create dataframe with best example per topic and topic description')
    df = pd.DataFrame()
    for topic_num, topic in enumerate(topic_list_human_readable):
        text_topic = text[text.topic_num == topic_num].sort_values(by=['score'], ascending=False).reset_index()
        frequency = len(text[text.topic_num == topic_num]) / len_original
        responses = len(text[text.topic_num == topic_num])
        if not text_topic.empty:
            representative_text = text_topic.iloc[0]['text']
            i = 1
            while (len([key for key in topic if key in representative_text.lower()]) < 2 or 'corona kills and has symptoms like' in representative_text.lower()):
                if i < len(text_topic):
                    representative_text = text_topic.iloc[0+i]['text']
                    i += 1
                else:
                    print("WARNING: no good example found")
                    representative_text = text_topic.iloc[0]['text']
                    break

            df = df.append(pd.Series({"topic number": int(topic_num),
                                      "example": representative_text,
                                      "keywords": ', '.join(topic),
                                      "frequency (%)": frequency * 100.,
                                      "number of responses": responses}), ignore_index=True)
    df = df.sort_values(by=['frequency (%)'], ascending=False)
    print(df.head())

    df = keywords_to_themes(df)

    print(df.head(10))
    df.to_csv('results_test.csv')

    # reformat data and push to google sheets
    data_to_upload = [['theme', 'example', 'keywords', 'frequency (%)', 'number of responses', 'topic number']] + \
                     df[['theme', 'example', 'keywords', 'frequency (%)', 'number of responses', 'topic number']].values.tolist()

    TargetRangeName = 'output!A:F'
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
    main()

