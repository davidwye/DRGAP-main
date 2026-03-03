from utils.file_utils import json_readb, jsonl_read
from config.config import datasetsPath
from datasets.winobias_run import Winobias
from datasets.winogender_run import Winogender


def get_coR_dev():

    return (json_readb(datasetsPath['winobias']['dev']),
            jsonl_read(datasetsPath['winogender']['dev']),
            json_readb(datasetsPath['GAP']['dev']),
            json_readb(datasetsPath['BUG']['dev']))

def get_coR_test():

    return (json_readb(datasetsPath['winobias']['test']),
            jsonl_read(datasetsPath['winogender']['test']),
            json_readb(datasetsPath['GAP']['test']),
            json_readb(datasetsPath['BUG']['test']))


def get_QA_dev():

    return (jsonl_read(datasetsPath['BBQ']['dev']),
            json_readb(datasetsPath['StereoSet']['dev']),
            json_readb(datasetsPath['UnQover']['dev']))

def get_QA_test():

    return (jsonl_read(datasetsPath['BBQ']['test']),
            json_readb(datasetsPath['StereoSet']['test']),
            json_readb(datasetsPath['UnQover']['test']))

def chooseFunc(datasetName, modelName, deviceId, isTest=False):

    if isTest:
        winobias, winogender, GAP, BUG = get_coR_dev()
        BBQ, StereoSet, UnQover = get_QA_dev()
    else:
        winobias, winogender, GAP, BUG = get_coR_test()
        BBQ, StereoSet, UnQover = get_QA_test()

    if datasetName == 'winobias':
        Obj = Winobias(modelName, deviceId)
        dataset = winobias
    elif datasetName == 'winogender':
        Obj = Winogender(modelName, deviceId)
        dataset = winogender
    elif datasetName == 'gap':
        Obj = GAP(modelName, deviceId)
        dataset = GAP
    elif datasetName == 'BUG':
        Obj = BUG(modelName, deviceId)
        dataset = BUG
    elif datasetName == 'BBQ':
        Obj = BBQ(modelName, deviceId)
        dataset = BBQ
    elif datasetName == 'StereoSet':
        Obj = StereoSet(modelName, deviceId)
        dataset = BBQ
    elif datasetName == 'UnQover':
        Obj = UnQover(modelName, deviceId)
        dataset = BBQ
    return Obj, dataset


def abstract_info(datasetName, faultExample):

    if datasetName.lower() == 'winobias' or datasetName.lower() == 'winogender':
        question = 'Identify the entity that the pronoun refers to the following sentence.'
        text = faultExample['sentence']
        answer = faultExample['answer']

    elif datasetName.lower() == 'gap':
        question = ('''Identify the entity that the pronoun '$PRONOUN' whose starting position is at the $OFFSETth
        character refers to in the following text.'''.replace("$PRONOUN", faultExample['pronoun'])
                    .replace("$OFFSET", faultExample['pronoun_offset']))
        text = faultExample['text']
        answer = faultExample['answer']

    elif datasetName.lower() == 'bug' or datasetName.lower() == 'bbq':
        question = faultExample['question']
        text = faultExample['text']
        answer = faultExample['answer']

    return question, text, answer