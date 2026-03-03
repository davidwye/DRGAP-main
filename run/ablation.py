'''
在winobias/winogedner/GAP/BUG 在llama3 分别去掉1/2/3/4模块 在dev上最佳的prompt 在test上的Acc Bias
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import reasonning_process
from utils.file_utils import json_readb, jsonl_write
from winobias_llama_run import Winobias
from datasets.datasets import winobias_run, winogender_run, gap_run, BUG_run
from winogender_llama_run import Winogender
from BUG_llama_run import BUG
from gap_llama_run import GAP
from datasets.datasets import BBQ,calculate_score


# 得到4(个数据集)*4(种模块配置)*5(个reasoning)的json
def get_reasonings(modelName, deviceId, datasetName, tag='0'):
    promptsDict = dict()
    res = reasonning_process.run(modelName, deviceId, datasetName, 'del1', CoTEnable=False)
    promptsDict['del1'] = res
    res = reasonning_process.run(modelName, deviceId, datasetName, 'del2', VerifyEnable=False)
    promptsDict['del2'] = res
    res = reasonning_process.run(modelName, deviceId, datasetName, 'del3', GenderIntEnable=False)
    promptsDict['del3'] = res
    res = reasonning_process.run(modelName, deviceId, datasetName, 'del4', RefineEnable=False)
    promptsDict['del4'] = res
    with open(f"DRs_result/{datasetName}_reasonings_ablation_{tag}.json", "w", encoding="utf-8") as file:
        json.dump(json.dumps(promptsDict, indent=4, ensure_ascii=False), file, indent=4, ensure_ascii=False)

def reasoning_datasets(modelName, deviceId, tag='0'):
    for datasetName in ['winobias', 'winogender', 'GAP', 'BUG']:
        get_reasonings(modelName, deviceId, datasetName)

# TODO 在dev上选择最佳的prompt
def run_on_dev(modelName, deviceId, datasetName, tag='0'):

    winobias_dev, winogender_dev, GAP_dev, BUG_dev = reasonning_process.get_coR_dev()
    if datasetName == 'winobias':
        M = Winobias(modelName, deviceId)
        res_deal = winobias_run.winobias_deal
        dev_set = winobias_dev
    elif datasetName == 'winogender':
        M = Winogender(modelName, deviceId)
        res_deal = winogender_run.winogender_deal
        dev_set = winogender_dev
    elif datasetName == 'gap':
        M = GAP(modelName, deviceId)
        res_deal = gap_run.gap_evaluation
        dev_set = GAP_dev
    elif datasetName == 'BUG':
        M = BUG(modelName, deviceId)
        res_deal = BUG_run.bug_evaluation
        dev_set = BUG_dev
    if datasetName == 'BBQ':
        M = BBQ(modelName, deviceId)
        res_deal = calculate_score
        dev_set = "datasets/BBQ_dev.jsonl"

    root = f'result_{modelName}'
    data = json_readb(f"DRs_result/{datasetName}_reasonings_ablation_{tag}.json")
    promptsDict = json.loads(data)

    for method in promptsDict:
        sub_prompts = promptsDict[method]

        for k, v in sub_prompts.items():
            v = v.replace('\n', '').replace('\\', '')
            if datasetName == 'BBQ':
                M.run(system=True, system_content=v, datafile=dev_set, resfile=f'{method}_{k}_{tag}')
                Acc, Bias, faultList = res_deal(os.path.join(root, f'{datasetName}_res_{method}_{k}_{tag}.jsonl'),modelName)
            else:
                M.run(system=True, system_content=v, data=dev_set, filename=f'{method}_{k}_{tag}', repTime=1)
                Acc, Bias, faultList = res_deal(os.path.join(root, f'{datasetName}_res_{method}_{k}_{tag}.json'))
            sub_prompts[k] = {"method": k,
                              "prompt": v,
                              "Acc": Acc,
                              "Bias": Bias}
            jsonl_write(f'DRs_result/{datasetName}_{modelName}_{method}_{tag}.jsonl', sub_prompts[k])
        promptsDict[method] = sub_prompts

    del M

    with open(f"DRs_result/{datasetName}_{modelName}_{tag}.json", "w", encoding="utf-8") as file:
        json.dump(json.dumps(promptsDict, indent=4, ensure_ascii=False), file, indent=4, ensure_ascii=False)

def select_the_best(modelName, deviceId, datasetName, tag):
    promptsDict = json.loads(json_readb(f"DRs_result/{datasetName}_{modelName}_{tag}.json"))
    bestPromptDict = dict()
    for method in promptsDict:
        sub_prompts = promptsDict[method]
        maxAcc = 0
        minBias = 1
        best_prompt = ""
        for k, v in sub_prompts.items():
            Acc = sub_prompts[k]['Acc']
            Bias = sub_prompts[k]['Bias']
            if Bias < minBias:
                minBias = Bias
                maxAcc = Acc
                best_prompt = sub_prompts[k]['prompt']
            elif Bias == minBias:
                if Acc >= maxAcc:
                    maxAcc = Acc
                    best_prompt = sub_prompts[k]['prompt']
        bestPromptDict[method] = {"prompt":best_prompt,
                                  "DevBias": minBias,
                                  "DevAcc": maxAcc}
    with open(f"DRs_result/{datasetName}_{modelName}_bestPrompt_{tag}.json", "w", encoding="utf-8") as file:
        json.dump(json.dumps(bestPromptDict, indent=4, ensure_ascii=False), file, indent=4, ensure_ascii=False)

# 在test上的结果
def run_on_test(modelName, deviceId, datasetName, tag):

    if datasetName == 'winobias':
        M = Winobias(modelName, deviceId)
        res_deal = winobias_run.winobias_deal
    elif datasetName == 'winogender':
        M = Winogender(modelName, deviceId)
        res_deal = winogender_run.winogender_deal
    elif datasetName == 'gap':
        M = GAP(modelName, deviceId)
        res_deal = gap_run.gap_evaluation
    elif datasetName == 'BUG':
        M = BUG(modelName, deviceId)
        res_deal = BUG_run.bug_evaluation
    if datasetName == 'BBQ':
        M = BBQ(modelName, deviceId)
        res_deal = calculate_score
        dev_set = "datasets/BBQ_dev.jsonl"

    root = f'result_{modelName}'
    data = json_readb(f"DRs_result/{datasetName}_{modelName}_bestPrompt_{tag}.json")
    bestPromptDict = json.loads(data)

    for k, v in bestPromptDict.items():
        print(v["prompt"])
        if datasetName == 'BBQ':
            M.run(system=True, system_content=v["prompt"], datafile="datasets/BBQ_test.jsonl", resfile=f'{k}_test_{tag}')
            Acc, Bias, faultList = res_deal(os.path.join(root, f'{datasetName}_res_{k}_test_{tag}.jsonl'), modelName)
        else:
            M.run(system=True, system_content=v["prompt"], filename=f'{k}_test_{tag}', repTime=1)
            Acc, Bias, faultList = res_deal(os.path.join(root, f'{datasetName}_res_{k}_test_{tag}.json'))
        bestPromptDict[k]["TestBias"] = Bias
        bestPromptDict[k]["TestAcc"] = Acc
        jsonl_write(f'DRs_result/{datasetName}_{modelName}_{k}_test_{tag}.jsonl', bestPromptDict[k])

    del M

    with open(f"DRs_result/{datasetName}_{modelName}_{tag}.json", "w", encoding="utf-8") as file:
        json.dump(json.dumps(bestPromptDict, indent=4, ensure_ascii=False), file, indent=4, ensure_ascii=False)


def get_mean_std(numbers):
    import statistics
    mean = statistics.mean(numbers)
    stdev = statistics.stdev(numbers)

    return mean, stdev

def result_compare(modelName, datasetName):

    if datasetName == 'winobias':
        res_deal = winobias_run.winobias_deal
    elif datasetName == 'winogender':
        res_deal = winogender_run.winogender_deal
    elif datasetName == 'gap':
        res_deal = gap_run.gap_evaluation
    elif datasetName == 'BUG':
        res_deal = BUG_run.bug_evaluation
    if datasetName == 'BBQ':
        res_deal = calculate_score

    res = dict()
    for method in [1,2,3,4]:
        AccList = []
        BiasList = []
        for tag in [0,0]:
            if datasetName == 'BBQ':
                Acc, Bias, faultList = res_deal(f'../res/Ablation/'
                                                'DRs_result/{datasetName}_res_del{method}_test_{tag}.jsonl', modelName)
            else:
                Acc, Bias, faultList = res_deal(f'../res/Ablation/DRs_result/{datasetName}_res_del{method}_test_{tag}.json')

            AccList.append(Acc)
            BiasList.append(Bias)
        accMean, accStd = get_mean_std(AccList)
        biasMean, biasStd = get_mean_std(BiasList)
        res[f'del{method}']=[accMean,biasMean]
    # print(res)
if __name__ == '__main__':

    # modelName = sys.argv[1]
    # deviceId = sys.argv[2]
    # datasetName = sys.argv[3]
    # tag = sys.argv[4]

    # get_reasonings(modelName, deviceId, datasetName, tag)
    # run_on_dev(modelName, deviceId, datasetName, tag)
    # select_the_best(modelName, deviceId, datasetName, tag)
    # run_on_test(modelName, deviceId, datasetName, tag)
    # result_compare('llama3', 'winobias')
    result_compare('llama3', 'winogender')
