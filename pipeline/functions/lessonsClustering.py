from gensim.models import TfidfModel
from gensim.models import LdaMulticore
import gensim.corpora as corpora
from gensim.utils import ClippedCorpus
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim import matutils
import pyLDAvis.gensim
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np 
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from DataFunctions import ElasticFunctions as ef

"""
    python lessonsClustering.py \
        --fine_tuning \
        --renew_tfidf \
        --number_of_topics 14 \
        --alpha 0.91 \
        --beta 0.91
"""

if __name__ == "__main__":

# Arguments
    parser = ArgumentParser()
    parser.add_argument("--fine_tuning", dest="fine_tuning", default=False, required=False, action='store_true')
    parser.add_argument("--renew_tfidf", dest="renew_tfidf", default=False, required=False, action='store_true')
    parser.add_argument("--renew_positions", dest="renew_positions", default=False, required=False, action='store_true')
    parser.add_argument("--number_of_topics", dest="number_of_topics", default=14, required=True, type=int)
    parser.add_argument("--alpha", dest="alpha", default=0.91, required=True, type=float)
    parser.add_argument("--beta", dest="beta", default=0.91, required=True, type=float)
    args = parser.parse_args()

# Credentials
    credentials = {
        # "ip_and_port": "52.163.240.214:9200",
        "ip_and_port": "52.230.8.63:9200",
        "username": "elastic",
        "password": "Welcometoerni!"
    }

    prodCredentials = {
        "ip_and_port": "52.163.240.214:9200",
        # "ip_and_port": "52.230.8.63:9200",
        "username": "elastic",
        "password": "Welcometoerni!"
    }

# Get lessons data from database

    df = ef.getLessons(credentials)

# Pre Processing
    lessonsData = df[df['isLesson'] == True]
    lessonsData = lessonsData[lessonsData['summary'] == lessonsData['summary']]
    lessonsData = lessonsData[0:20]
    raw_paragraphs = lessonsData['paragraph']
    urls = lessonsData['urlToFile']
    raw_sentences = raw_paragraphs
    ids = lessonsData['_id']
    referenceIds = lessonsData['referenceId']
    sentences = [line.split(' ') for line in raw_sentences]    
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'äô', 'äù', 'äì'])
    words_to_remove = ['iii', 'project']

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def remove_words(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in words_to_remove] for doc in texts]

    def remove_word_length_2(texts):
        allSentences = []
        for doc in texts:
            newWords = []
            for word in doc:
                if len(word) > 2:
                    newWords.append(word)
            allSentences.append(newWords)
        return allSentences

    def replace_adb_special_characters(texts):
        return [[word.replace('‚Äôs', "'s ").replace('O‚ÄôSmach', "0").replace('äù', "").replace('äô', "").replace('äì', "") for word in doc] for doc in texts]

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    sentences = replace_adb_special_characters(sentences)
    data_words_nostops = remove_stopwords(sentences)
    lemmatizer = WordNetLemmatizer()
    lemmatized_output = []
    for paragraph in data_words_nostops:
        lemmatized_output.append([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in paragraph])
    sentences = remove_words(lemmatized_output)
    sentences_no_length_2 = remove_word_length_2(sentences)
    sentences = sentences_no_length_2

    id2word = corpora.Dictionary(sentences)
    texts = sentences
    corpus = [id2word.doc2bow(text) for text in texts]

    def compute_coherence_values(corpus, dictionary, k, a, b):
        lda_model = LdaMulticore(corpus=corpus,
                                id2word=id2word,
                                num_topics=k, 
                                random_state=100,
                                chunksize=100,
                                passes=10,
                                alpha=a,
                                eta=b,
                                per_word_topics=True)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=sentences, dictionary=id2word, coherence='c_v')
        return coherence_model_lda.get_coherence()

# Fine Tuning
    if args.fine_tuning:
        import numpy as np
        import pandas as pd
        import tqdm
        grid = {}
        grid['Validation_Set'] = {}

        # Topics range
        min_topics = 2
        max_topics = 15
        step_size = 1
        topics_range = range(min_topics, max_topics, step_size)

        # Alpha parameter
        alpha = list(np.arange(0.01, 1, 0.3))
        alpha.append('symmetric')
        alpha.append('asymmetric')

        # Beta parameter
        beta = list(np.arange(0.01, 1, 0.3))
        beta.append('symmetric')

        # Validation sets
        num_of_docs = len(corpus)
        corpus_sets = [
        #                ClippedCorpus(corpus, int(num_of_docs*0.25)), 
                    ClippedCorpus(corpus, int(num_of_docs*0.5)), 
        #                ClippedCorpus(corpus, int(num_of_docs*0.75)), 
                    corpus
                    ]
        corpus_title = [
                        '25% Corpus'
                        '50% Corpus', 
                        '75% Corpus'
                        '100% Corpus'
                    ]
        model_results = {'Validation_Set': [],
                        'Topics': [],
                        'Alpha': [],
                        'Beta': [],
                        'Coherence': []
                        }
        if 1 == 1:
            pbar = tqdm.tqdm(total=2000)
            for i in range(len(corpus_sets)):
                for k in topics_range:
                    for a in alpha:
                        for b in beta:
                            cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                        k=k, a=a, b=b)
                            model_results['Validation_Set'].append(corpus_title[i])
                            model_results['Topics'].append(k)
                            model_results['Alpha'].append(a)
                            model_results['Beta'].append(b)
                            model_results['Coherence'].append(cv)
                            pbar.update(1)
                            # TODO: Save to elasticsearch
                            # pd.DataFrame(model_results).to_csv(oldDataFileName + '-fine-tuning-.csv', index=False)

# Run LDA model
    lda_model = LdaMulticore(corpus=corpus,
                        id2word=id2word,
                        num_topics=args.number_of_topics, 
                        random_state=200,
                        chunksize=100,
                        passes=10,
                        alpha=args.alpha,
                        eta=args.beta,
                        per_word_topics=True)

# Save TFIDF model
    if args.renew_tfidf:
        tfidf = TfidfModel(corpus, smartirs='ntc')
        tfidf_corpus = []
        for doc in corpus:
            tfidf_corpus.append(tfidf[doc])
        tfidf_mat = matutils.corpus2dense(tfidf_corpus, num_terms=len(id2word.token2id))
        tfidf_mat_transpose = tfidf_mat.transpose()
        dfTFIDF=pd.DataFrame(data=tfidf_mat_transpose[0:,0:],
                            index=[i for i in range(tfidf_mat_transpose.shape[0])],
                            columns=[''+str(i) for i in range(tfidf_mat_transpose.shape[1])])
        dfTFIDF['id'] = ids.tolist()

        ef.deleteIndex(credentials, "tfidf")
        ef.saveTFIDF(credentials, dfTFIDF)

# Keyword weights
    x=lda_model.show_topics(num_topics=args.number_of_topics, num_words=50,formatted=False)
    keywordWeights = []
    topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
    for tp in x:
        words = []
        weights = []
        for pair in tp[1]:
            words.append(pair[0])
            weights.append(int(pair[1]*10000))
        keywordWeights.append(weights)

# Top topics per paragraph
    df = pd.DataFrame()
    df['referenceId'] = referenceIds
    df['paragraph'] = raw_paragraphs
    topicNumbers = []
    for c in range(len(corpus)):
        maxProbability = 0
        indexOfMax = 0
        topTopics = []
        topTopicProbabilities = []
        lda_model.get_document_topics(corpus[c])
        for topicNumber in lda_model.get_document_topics(corpus[c]):
            topTopics.append(topicNumber[0])
            topTopicProbabilities.append(topicNumber[1])
        topTopicsSorted = [x for _, x in sorted(zip(topTopicProbabilities, topTopics), reverse=True)]
        topicNumbers.append(topTopicsSorted)
    df['topTopics'] = topicNumbers

# Most probable topic per paragraph
    topTopics = []
    for index, row in df.iterrows():
        if(row['topTopics']):
            topTopics.append(row['topTopics'][0])
        else:
            topTopics.append(-1)
    df['topic'] = topTopics

# Frequencies of topic keywords and number of PCRs per topic
    topics = pd.DataFrame()
    topicKeywords = []
    allKeywords = []
    topicIds = []
    for topic, words in topics_words:
        allKeywords.append(words)
        topicIds.append(topic)
    topics['key'] = topicIds
    topics['keywords'] = allKeywords
    topics['oldFrequencies'] = [ [0] * len(keywords) for keywords in allKeywords]
    topics['numberOfLessons'] = 0
    topics['PCRs'] = [[]  for i in range(len(topics))]
    topics['numberOfPCRs'] = 0

    for sentenceTopicNumbers, sentenceURL in zip(topicNumbers, urls):
        for topicNumber in sentenceTopicNumbers:
            topics.at[topicNumber, 'numberOfLessons'] = topics.at[topicNumber, 'numberOfLessons'] + 1
            topics.at[topicNumber, 'PCRs'].append(sentenceURL)
    for index, row in topics.iterrows():
        topics.at[index, 'numberOfPCRs'] = len(set(topics.at[index, 'PCRs']))
    topics = topics.drop(columns=['PCRs'])

# Frequencies of words per sentence per topic
    topics['oldFrequencies'] = [ [0] * len(keywords) for keywords in allKeywords]
    for index, row in topics.iterrows():
        topicNumber = topics.at[index, 'key']
        topicKeywords = topics.at[index, 'keywords']
        topicKeywordsFrequencies = topics.at[index, 'oldFrequencies']
        for sentence, sentenceTopicNumbers in zip(sentences, topicNumbers):
            for sentenceTopicNumber in sentenceTopicNumbers:
                if topicNumber == sentenceTopicNumber:
                    for word in sentence:
                        if word in topicKeywords:
                            indexOfWord = topicKeywords.index(word)
                            topicKeywordsFrequencies[indexOfWord] = topicKeywordsFrequencies[indexOfWord] + 1
        topics.at[index, 'oldFrequencies'] = topicKeywordsFrequencies
    topics['frequencies'] = keywordWeights
    
# Top word per topic
    topicTopWords = []
    for index, row in topics.iterrows():
        topicTopWords.append(row['keywords'][0])
    topics['topWord'] = topicTopWords

# Adjacent topics
    # pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    topics['x'] = 1.0
    topics['y'] = 1.0
    for topic, x in zip(list(vis.topic_coordinates.index), list(vis.topic_coordinates.x)):
        topics.at[topic, 'x'] = float(x)
    for topic, y in zip(list(vis.topic_coordinates.index), list(vis.topic_coordinates.y)):
        topics.at[topic, 'y'] = float(y)

    import math  
    def calculateDistance(x1,y1,x2,y2):  
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
        return dist  

    distanceMatrix = []
    allDistances = []
    c1 = 0
    topicsX = topics['x'].tolist()
    topicsY = topics['y'].tolist()
    for tx1, ty1 in zip(topicsX, topicsY):
        distances = []
        for tx2, ty2 in zip(topicsX, topicsY):
            distance = calculateDistance(tx1, ty1, tx2, ty2)
            if not distance:
                distance = 999
            else:
                allDistances.append(distance)
            distances.append(distance)
        distanceMatrix.append(distances)
        c1 = c1 + 1
    
    percentile20 = np.percentile(allDistances, 20)
    numberOfAdjacent = 0
    numberOfNodes = len(distanceMatrix)
    allAdjacentTopics = []
    for distances in distanceMatrix:
        adjacentTopics = []
        for index, distance in zip(range(len(distances)), distances):
            if distance <= percentile20:
                adjacentTopics.append(index)
        allAdjacentTopics.append(adjacentTopics)
        numberOfAdjacent = numberOfAdjacent + len(adjacentTopics)
    numberOfAdjacent = numberOfAdjacent/2
    pairs = []
    for index, adjacentTopicList in zip(range(len(allAdjacentTopics)), allAdjacentTopics):
        for adjacentTopic in adjacentTopicList:
            pairs.append(sorted([index, adjacentTopic]))
    pairs.sort()
    dedupedPairs = list(pairs for pairs, _ in itertools.groupby(pairs))
    topWordPairs = []
    for pair in dedupedPairs:
        topWordPairs.append([topicTopWords[pair[0]], topicTopWords[pair[1]]])
    topics['adjacentTopics'] = allAdjacentTopics

# Save topics data
    ef.deleteIndex(credentials, "topics")
    ef.saveTopics(credentials, topics)

# Lesson strength
    maxLessonStrength = topics['numberOfPCRs'].sum()
    lessonStrengths = []
    for index, row in df.iterrows():
        topicNumbers = row['topTopics']
        lessonStrength = 0
        for topicNumber in topicNumbers:
            lessonStrength = lessonStrength + topics.at[topicNumber, 'numberOfPCRs']
        lessonStrengths.append(lessonStrength/maxLessonStrength)
    df['lessonStrength'] = lessonStrengths
    # dfLessonStrength = df[['_id', 'Lesson Strength','topic', 'topTopics']]

# TODO: Save lesson topics, lesson strengths
    # ef.updateLessons(credentials, df)