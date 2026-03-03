import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from reasonning_process import reasoning_generation
from datasets.datasets import chooseFunc
from utils.file_utils import json_readb
from config.config import root


def run_on_dev(modelName, deviceId, datasetName):

    Obj, res_evaluate, dataset = chooseFunc(datasetName, modelName, deviceId)

    data = json_readb(os.path.join(root, f"{datasetName}_reasonings.json"))
    promptsDict = json.loads(data)

    for method in promptsDict:
        sub_prompts = promptsDict[method]

        for k, v in sub_prompts.items():
            Obj.run(system=True, system_content=v, data=dataset, filename=f'{method}_{k}')
            Acc, Bias, _ = res_evaluate(os.path.join(root, f'{datasetName}_res_{method}_{k}.json'), modelName)

            sub_prompts[k] = {"method": k,
                              "prompt": v,
                              "Acc": Acc,
                              "Bias": Bias}

        promptsDict[method] = sub_prompts

    del Obj

    return promptsDict

def select_the_best(promptsDict, modelName, datasetName):

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

    with open(os.path.join(root, 'bestPrompt', f"{datasetName}_{modelName}.json"), "w", encoding="utf-8") as file:
        json.dump(json.dumps(bestPromptDict, indent=4, ensure_ascii=False), file, indent=4, ensure_ascii=False)


def run_on_test(modelName, deviceId, datasetName):

    Obj, res_evaluate, dataset = chooseFunc(datasetName, modelName, deviceId, isTest=True)

    data = json_readb(os.path.join(root, 'bestPrompt', f"{datasetName}_{modelName}.json"))
    bestPromptDict = json.loads(data)

    for k, v in bestPromptDict.items():
        Obj.run(system=True, system_content=v["prompt"], filename=f'{k}', repTime=1)
        Acc, Bias, _ = res_evaluate(os.path.join(root, 'testRes', f'{datasetName}_res_{k}'))
        bestPromptDict[k]["TestBias"] = Bias
        bestPromptDict[k]["TestAcc"] = Acc

    del Obj

    with open(os.path.join(root, 'testRes', f"{datasetName}_{modelName}.json"), "w", encoding="utf-8") as file:
        json.dump(json.dumps(bestPromptDict, indent=4, ensure_ascii=False), file, indent=4, ensure_ascii=False)

def evaluate(modelName, datasetName):
    pass

def run(modelName, deviceId, datasetName):

    reasoning_generation(modelName, deviceId, datasetName)
    select_the_best(run_on_dev(modelName, deviceId, datasetName), modelName, datasetName)
    run_on_test(modelName, deviceId, datasetName)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--function", type=str, default="reasoning")
    parser.add_argument("--modelName", type=str, default="llama3")
    parser.add_argument("--deviceId", type=int, default=0)
    parser.add_argument("--datasetName", type=str, default="winobias")
    args = parser.parse_args()

    run(args.modelName, args.deviceId, args.datasetName)