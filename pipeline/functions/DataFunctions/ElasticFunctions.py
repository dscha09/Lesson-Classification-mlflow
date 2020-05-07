import pandas as pd
import datetime
from elasticsearch import Elasticsearch, helpers
from pandasticsearch import Select
from es_pandas import es_pandas
from espandas import Espandas
from argparse import ArgumentParser


"""
    credentials = {
        "ip_and_port": "52.163.240.214:9200",     # production
        "ip_and_port": "52.230.8.63:9200",         # staging
        "username": "elastic",
        "password": "Welcometoerni!"
    }
"""


"""
    Description: Function to delete an index
    Returns: None
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> ef.deleteIndex(credentials, index)
"""
def deleteIndex(credentials, index):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    es.indices.delete(index=index, ignore=[400, 404])


"""
    Description: Function to query all sentences
    Returns: dataframe with all the sentences
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> df = ef.getLessons(credentials)
"""
def getSentences(credentials):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    sentencesDF = pd.DataFrame()
    data = es.search(index="sentences", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if sentencesDF.empty:
            sentencesDF = Select.from_dict(data).to_pandas()
        else:
            sentencesDF = sentencesDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return sentencesDF


def getBaseClassification(credentials):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    baseClassificationDF = pd.DataFrame()
    data = es.search(index="base-classification", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if baseClassificationDF.empty:
            baseClassificationDF = Select.from_dict(data).to_pandas()
        else:
            baseClassificationDF = baseClassificationDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return baseClassificationDF


"""
    Description: Function to query all lessons
    Returns: dataframe with all the sentences
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> df = ef.getSentences(credentials)
"""
def getLessons(credentials):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
        "query": {
            "term": {
                "isLesson": {
                    "value": True,
                    "boost": 1.0
                }
            }
        }
    }
    lessonsDF = pd.DataFrame()
    data = es.search(index="sentences", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if lessonsDF.empty:
            lessonsDF = Select.from_dict(data).to_pandas()
        else:
            lessonsDF = lessonsDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return lessonsDF


"""
    Description: Function to update lessons
    Returns: None
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> ef.updateLessons(credentials, lessonsDF)
"""
def updateSentences(credentials, updatedDF):
    sentencesDF = getSentences(credentials)
    sentencesDF["id"] = sentencesDF["id"].astype('str')
    sentencesDF = sentencesDF.set_index("id")
    updatedDF["id"] = updatedDF["id"].astype('str')
    updatedDF = updatedDF.set_index("id")
    sentencesDF.update(updatedDF)
    sentencesDF.reset_index(inplace=True)
    if "_index" in sentencesDF.columns:
        sentencesDF = sentencesDF.drop(columns=["_index"])
    if "_type" in sentencesDF.columns:
        sentencesDF = sentencesDF.drop(columns=["_type"])
    if "_id" in sentencesDF.columns:
        sentencesDF = sentencesDF.drop(columns=["_id"])
    if "_score" in sentencesDF.columns:
        sentencesDF = sentencesDF.drop(columns=["_score"])
    sentencesDF["id"] = sentencesDF["id"].astype('str')
    deleteIndex(credentials, "sentences")
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    ep.init_es_tmpl(sentencesDF, "sentences")
    ep.to_es(sentencesDF, "sentences", doc_type="sentences")


"""
    Description: Function to update TFIDF matrix
    Returns: None
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> ef.deleteIndex(credentials, tfidfDF)
"""
def saveTFIDF(credentials, dfTFIDF):
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    ep.init_es_tmpl(dfTFIDF, "tfidf")
    ep.to_es(dfTFIDF, "tfidf", doc_type="tfidf")


"""
    Description: Function to query TFIDF matrix
    Returns: TFIDF dataframe
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> df = ef.getTFIDF(credentials)
"""
def getTFIDF(credentials):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    dfTFIDF = pd.DataFrame()
    data = es.search(index="tfidf", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if dfTFIDF.empty:
            dfTFIDF = Select.from_dict(data).to_pandas()
        else:
            dfTFIDF = dfTFIDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return dfTFIDF


"""
    Description: Function to query all topics
    Returns: dataframe with all the topics
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> df = ef.getTopics(credentials)
"""
def getTopics(credentials):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    topicsDF = pd.DataFrame()
    data = es.search(index="topics", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if topicsDF.empty:
            topicsDF = Select.from_dict(data).to_pandas()
        else:
            topicsDF = topicsDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return topicsDF


"""
    Description: Function to update topics
    Returns: None
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> ef.deleteIndex(credentials, tfidfDF)
"""
def saveTopics(credentials, topicsDF):
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    if "_index" in topicsDF.columns:
        topicsDF = topicsDF.drop(columns=["_index"])
    if "_type" in topicsDF.columns:
        topicsDF = topicsDF.drop(columns=["_type"])
    if "_id" in topicsDF.columns:
        topicsDF = topicsDF.drop(columns=["_id"])
    if "_score" in topicsDF.columns:
        topicsDF = topicsDF.drop(columns=["_score"])
    ep.init_es_tmpl(topicsDF, "topics")
    ep.to_es(topicsDF, "topics", doc_type="topics")


# In progress
"""
    Description: Function to update base classifications with the new annotations
    Returns: None
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> ef.updateBaseClassification(credentials)
"""
def updateBaseClassification(credentials):
    annotatedDF = getAnnotatedSentences(credentials)
    baseDF = getBaseClassification(credentials)
    existingIds = baseDF["sentencesId"]
    newAnnotatedDF = annotatedDF.loc[~annotatedDF["id"].isin(existingIds)]
    newAnnotatedDF = newAnnotatedDF[["id", "paragraph", "isLesson"]]
    newAnnotatedDF = newAnnotatedDF.rename(columns={"id": "sentencesId"})
    newAnnotatedDF["source"] = "annotation"
    newIdStart = max(baseDF["id"].astype('int').tolist()) + 1
    newAnnotatedDF["id"] = range(newIdStart, newIdStart + len(newAnnotatedDF))
    newAnnotatedDF["id"] = newAnnotatedDF["id"].astype('str')
    if "_index" in baseDF.columns:
        baseDF = baseDF.drop(columns=["_index"])
    if "_type" in baseDF.columns:
        baseDF = baseDF.drop(columns=["_type"])
    if "_id" in baseDF.columns:
        baseDF = baseDF.drop(columns=["_id"])
    if "_score" in baseDF.columns:
        baseDF = baseDF.drop(columns=["_score"])
    baseDF = baseDF.append(newAnnotatedDF)
    deleteIndex(credentials, "tmp")
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    ep.init_es_tmpl(baseDF, "tmp")
    ep.to_es(baseDF, "tmp", doc_type="tmp")


import pandas as pd
import datetime
from elasticsearch import Elasticsearch, helpers
from pandasticsearch import Select
from es_pandas import es_pandas
from espandas import Espandas
from argparse import ArgumentParser


"""
    credentials = {
        "ip_and_port": "52.163.240.214:9200",     # production
        "ip_and_port": "52.230.8.63:9200",         # staging
        "username": "elastic",
        "password": "Welcometoerni!"
    }
"""


"""
    Description: Function to delete an index
    Returns: None
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> ef.deleteIndex(credentials, index)
"""
def deleteIndex(credentials, index):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    es.indices.delete(index=index, ignore=[400, 404])


"""
    Description: Function to query all sentences
    Returns: dataframe with all the sentences
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> df = ef.getLessons(credentials)
"""
def getSentences(credentials):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    sentencesDF = pd.DataFrame()
    data = es.search(index="sentences", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if sentencesDF.empty:
            sentencesDF = Select.from_dict(data).to_pandas()
        else:
            sentencesDF = sentencesDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return sentencesDF


"""
    Description: Function to query all lessons
    Returns: dataframe with all the sentences
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> df = ef.getSentences(credentials)
"""
def getLessons(credentials):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
        "query": {
            "term": {
                "isLesson": {
                    "value": True,
                    "boost": 1.0
                }
            }
        }
    }
    lessonsDF = pd.DataFrame()
    data = es.search(index="sentences", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if lessonsDF.empty:
            lessonsDF = Select.from_dict(data).to_pandas()
        else:
            lessonsDF = lessonsDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return lessonsDF


"""
    Description: Function to update lessons
    Returns: None
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> ef.updateLessons(credentials, lessonsDF)
"""
def updateSentences(credentials, updatedDF):
    sentencesDF = getSentences(credentials)
    sentencesDF["id"] = sentencesDF["id"].astype('str')
    sentencesDF = sentencesDF.set_index("id")
    updatedDF["id"] = updatedDF["id"].astype('str')
    updatedDF = updatedDF.set_index("id")
    sentencesDF.update(updatedDF)
    sentencesDF.reset_index(inplace=True)
    if "_index" in sentencesDF.columns:
        sentencesDF = sentencesDF.drop(columns=["_index"])
    if "_type" in sentencesDF.columns:
        sentencesDF = sentencesDF.drop(columns=["_type"])
    if "_id" in sentencesDF.columns:
        sentencesDF = sentencesDF.drop(columns=["_id"])
    if "_score" in sentencesDF.columns:
        sentencesDF = sentencesDF.drop(columns=["_score"])
    sentencesDF["id"] = sentencesDF["id"].astype('str')
    deleteIndex(credentials, "sentences")
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    ep.init_es_tmpl(sentencesDF, "sentences")
    ep.to_es(sentencesDF, "sentences", doc_type="sentences")


"""
    Description: Function to update TFIDF matrix
    Returns: None
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> ef.deleteIndex(credentials, tfidfDF)
"""
def saveTFIDF(credentials, dfTFIDF):
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    ep.init_es_tmpl(dfTFIDF, "tfidf")
    ep.to_es(dfTFIDF, "tfidf", doc_type="tfidf")


"""
    Description: Function to query TFIDF matrix
    Returns: TFIDF dataframe
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> df = ef.getTFIDF(credentials)
"""
def getTFIDF(credentials):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    dfTFIDF = pd.DataFrame()
    data = es.search(index="tfidf", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if dfTFIDF.empty:
            dfTFIDF = Select.from_dict(data).to_pandas()
        else:
            dfTFIDF = dfTFIDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return dfTFIDF


"""
    Description: Function to query all topics
    Returns: dataframe with all the topics
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> df = ef.getTopics(credentials)
"""
def getTopics(credentials):
    es = Elasticsearch(['http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"]], timeout=600)
    doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
        }
    }
    topicsDF = pd.DataFrame()
    data = es.search(index="topics", body=doc, scroll='1m')
    scrollId = data['_scroll_id']
    scrollSize = len(data['hits']['hits'])
    while scrollSize > 0:
        if topicsDF.empty:
            topicsDF = Select.from_dict(data).to_pandas()
        else:
            topicsDF = topicsDF.append(Select.from_dict(data).to_pandas())
        data = es.scroll(scroll_id = scrollId, scroll = '1m')
        scrollId = data['_scroll_id']
        scrollSize = len(data['hits']['hits'])
    return topicsDF


"""
    Description: Function to update topics
    Returns: None
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> ef.deleteIndex(credentials, tfidfDF)
"""
def saveTopics(credentials, topicsDF):
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    if "_index" in topicsDF.columns:
        topicsDF = topicsDF.drop(columns=["_index"])
    if "_type" in topicsDF.columns:
        topicsDF = topicsDF.drop(columns=["_type"])
    if "_id" in topicsDF.columns:
        topicsDF = topicsDF.drop(columns=["_id"])
    if "_score" in topicsDF.columns:
        topicsDF = topicsDF.drop(columns=["_score"])
    ep.init_es_tmpl(topicsDF, "topics")
    ep.to_es(topicsDF, "topics", doc_type="topics")


# In progress
"""
    Description: Function to update base classifications with the new annotations
    Returns: None
    Usage:
    >>> from DataFunctions import ElasticFunctions as ef
    >>> ef.updateBaseClassification(credentials)
"""
def updateBaseClassification(credentials):
    annotatedDF = getAnnotatedSentences(credentials)
    baseDF = getBaseClassification(credentials)
    existingIds = baseDF["sentencesId"]
    newAnnotatedDF = annotatedDF.loc[~annotatedDF["id"].isin(existingIds)]
    newAnnotatedDF = newAnnotatedDF[["id", "paragraph", "isLesson"]]
    newAnnotatedDF = newAnnotatedDF.rename(columns={"id": "sentencesId"})
    newAnnotatedDF["source"] = "annotation"
    newIdStart = max(baseDF["id"].astype('int').tolist()) + 1
    newAnnotatedDF["id"] = range(newIdStart, newIdStart + len(newAnnotatedDF))
    newAnnotatedDF["id"] = newAnnotatedDF["id"].astype('str')
    if "_index" in baseDF.columns:
        baseDF = baseDF.drop(columns=["_index"])
    if "_type" in baseDF.columns:
        baseDF = baseDF.drop(columns=["_type"])
    if "_id" in baseDF.columns:
        baseDF = baseDF.drop(columns=["_id"])
    if "_score" in baseDF.columns:
        baseDF = baseDF.drop(columns=["_score"])
    baseDF = baseDF.append(newAnnotatedDF)
    deleteIndex(credentials, "tmp")
    ep = es_pandas('http://' + credentials["username"] + ':' + credentials["password"] + '@' + credentials["ip_and_port"])
    ep.init_es_tmpl(baseDF, "tmp")
    ep.to_es(baseDF, "tmp", doc_type="tmp")


