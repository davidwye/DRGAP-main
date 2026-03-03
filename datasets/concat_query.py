from datasets import winobias_run,winogender_run,gap_run,BUG_run,BBQ_run,StereoSet_run,UnQover_run
import json
def get_data(datasetName, filePath):
    if datasetName == "winobias":
        data = winobias_run.get_all_prompt(filePath)
    elif datasetName == "winogender":
        data = winogender_run.get_all_prompt(filePath)
    elif datasetName == "BBQ":
        data = BBQ_run.get_all_data(filePath)
    return data

def save_res(datasetName, data, resLists, timeList, modelName, filename):
    if datasetName == "winobias":
        for idx in range(len(data)):
            data[list(data.keys())[idx]]['ans'] =resLists[idx]
            data[list(data.keys())[idx]]['use_time'] = timeList[idx]
        with open('{}.json'.format(filename), 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    elif datasetName == "winogender":
        result = dict()
        for idx in range(len(data)):
            each = data[list(data.keys())[idx]]
            each['ans'] = resLists[idx]
            each['use_time'] = timeList[idx]
            result['{}_{}_{}_{}'.format(each['occupation'],each['participant'],each['answer'], each['pronoun'])] = each
        with open('{}.json'.format(filename), 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)

    elif datasetName == "BBQ":
        for idx in range(len(data)):
            each = data[list(data.keys())[idx]]
            each[f'pre_{modelName}'] = resLists[idx]
            each['use_time'] = timeList[idx]
        with open(f'{filename}.jsonl', 'a') as file:
            json.dump(each, file)
            file.write("\n")

    return data

def run(datasetName, data, each):
    if datasetName == "winobias":
        question = "Which entity that the pronoun refers to the following sentence."
        background = data[each]['sentence']
    elif  datasetName == "winogender":
        question = "Which entity that the pronoun refers to the following sentence."
        background = each['sentence']
    elif datasetName == "BBQ":
        question = f"{each['question']} 0.[{each['ans0']}] 1.[{each['ans1']}] 2.[{each['ans2']}]"
        background = each['context']
    return background, question
