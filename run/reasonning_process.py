import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import call_gpt
import json
import random
from datasets.datasets import chooseFunc, abstract_info
from config.config import DRGAPPromptTemplates, modelMap

class ReasoningGeneration():

    def __init__(self):
        self.system_prompt = DRGAPPromptTemplates['system']
        self.initial = DRGAPPromptTemplates['initial']
        self.verification = DRGAPPromptTemplates['verification']
        self.filtering = DRGAPPromptTemplates['filtering']
        self.refinement = DRGAPPromptTemplates['refinement']
        self.template = DRGAPPromptTemplates['template']
        self.model = modelMap['GPT4']

    def initial_reasoning(self, question, text, answer):
        prompt = self.initial.replace("$QUESTION", question).replace("$TEXT", text).replace("$ANSWER", answer)
        res = call_gpt(model=self.model, user_content=prompt, system=True, system_content=self.system_prompt)
        return res


    def verification_reasoning(self, question, text, answer, reasoning):
        prompt = self.verification.replace("$QUESTION", question).replace("$TEXT", text).replace("$REASONING", reasoning)
        res = call_gpt(model=self.model, user_content=prompt, system=True, system_content=self.system_prompt)
        return res

    def filtering_reasoning(self, question, text, answer, reasoning):
        prompt = self.filtering.replace("$QUESTION", question).replace("$TEXT", text).replace("$REASONING", reasoning)
        res = call_gpt(model=self.model, user_content=prompt, system=True, system_content=self.system_prompt)
        return res

    def refinement_reasoning(self, question, text, answer, reasoning):
        prompt = self.refinement.replace("$QUESTION", question).replace("$TEXT", text).replace("$REASONING", reasoning)
        res = call_gpt(model=self.model, user_content=prompt, system=True, system_content=self.system_prompt)
        return res

    def construct_system_content(self, question, text, answer, reasoning):
        return self.template.replace("$QUESTION", question).replace("$TEXT", text).replace("$REASONING", reasoning).replace("$ANSWER", answer)

def random_example(dataList):

    random_number = random.randint(0, len(dataList) - 1)
    faultExample = dataList[random_number]

    return faultExample

def random_fault_example(root, datasetName, Obj, res_evaluate, dataset):

    if not os.path.isfile(os.path.join(root, f'{datasetName}_res_dev.json')):
        Obj.run(data=dataset, filename='dev_baseline')
    _, _, originalFaultList = res_evaluate(os.path.join(root, f'{datasetName}_res_dev.json'))

    if not os.path.isfile(os.path.join(root, f'{datasetName}_res_dev_baseline.json')):
        Obj.run(model=modelMap['GPT4'], data=dataset, filename='dev_baseline')
    _, _, referenceFaultList = res_evaluate(os.path.join(root, f'{datasetName}_res_dev_baseline.json'))

    faultList = [item for item in originalFaultList if item not in referenceFaultList]

    if len(faultList) == 0:
        raise RuntimeError("No examples meet the condition.")

    random_number = random.randint(0, len(faultList) - 1)
    faultExample = faultList[random_number]

    return faultExample

def reasoning_generation(modelName, deviceId, datasetName, VerifyEnable=True, GenderIntEnable=True, RefineEnable=True):

    promptDict = dict()
    root = f'result_{modelName}'

    Obj, res_evaluate, dataset = chooseFunc(datasetName, modelName, deviceId)
    faultExample = random_fault_example(root, datasetName, Obj, res_evaluate, dataset) if datasetName not in ['StereoSet', 'UnQover'] else random_example(dataset)
    question, text, answer = abstract_info(datasetName, faultExample)

    # pipeline
    RGobj = ReasoningGeneration()
    reasoning = RGobj.initial_reasoning(question, text, answer)
    if VerifyEnable:
        reasoning = RGobj.verification_reasoning(question, text, answer, reasoning)
        promptDict['Verification'] = RGobj.construct_system_content(question, text, answer, reasoning)
    if GenderIntEnable:
        reasoning = RGobj.filtering_reasoning(question, text, answer, reasoning)
        promptDict['GenderInd'] = RGobj.construct_system_content(question, text, answer, reasoning)
    if RefineEnable:
        for i in range(3):
            reasoning = RGobj.refinement_reasoning(question, text, answer, reasoning)
            promptDict[f'Refine{i}'] = RGobj.construct_system_content(question, text, answer, reasoning)
    with open(os.path.join(root,f"{datasetName}_reasonings.json"), "w", encoding="utf-8") as file:
        json.dump(json.dumps(promptDict), file, indent=4, ensure_ascii=False)

    del Obj, RGobj

    return promptDict

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--modelName", type=str, default="llama3")
    parser.add_argument("--deviceId", type=int, default=0)
    parser.add_argument("--datasetName", type=str, default="winobias")
    parser.add_argument("--VerifyEnable", type=bool, default=True)
    parser.add_argument("--GenderIntEnable", type=bool, default=True)
    parser.add_argument("--RefineEnable", ype=bool, default=True)
    args = parser.parse_args()

    reasoning_generation(args.modelName, args.deviceId, args.datasetName, args.VerifyEnable, args.GenderIntEnable, args.RefineEnable)
