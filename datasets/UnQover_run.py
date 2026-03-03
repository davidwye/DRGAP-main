import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config.config import modelMap, QueryPromptTemplates
from models import init_model, get_probability
from utils.file_utils import jsonl_read, json_readb, jsonl_write

class UnQover():

    def __init__(self, modelName, deviceId):
        self.modelName = modelName
        self.DEVICE = torch.device(f"cuda:{deviceId}" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = init_model(modelMap[modelName], self.DEVICE)

    def get_all_data(self, filename):
        data = json_readb(filename)
        return data
    def get_prep(self, context, q, answers, system=False, system_content='', ):
        user_content = f'Question: {context} {q}'
        if system:
            prompt = (QueryPromptTemplates[f"{self.modelName}_system"]
                      .replace('$USERCONTENT', user_content)
                      .replace('$SYSTEMCONTENT', system_content))
        else:
            prompt = QueryPromptTemplates[f"{self.modelName}"].replace('$USERCONTENT', user_content)
        output = get_probability(prompt, self.tokenizer, self.model, self.DEVICE, answers)
        return output

    def run(self, datafile, output, system, system_content):
        datas = self.get_all_data(datafile)
        for item in datas:
            data = datas[item]
            context0 = data['context0']
            context1 = data['context1']
            q0 = data['q0']['question']
            q1 = data['q1']['question']
            m_name = data['q0']['ans0']['text']
            f_name = data['q0']['ans1']['text']
            [data['q0']['ans0']['pred0'],data['q0']['ans1']['pred0']] = self.get_prep(context0, q0, [m_name, f_name],
                                                                                     system, system_content)
            [data['q1']['ans0']['pred0'],data['q1']['ans1']['pred0']] = self.get_prep(context0, q1, [m_name, f_name],
                                                                                     system, system_content)
            [data['q0']['ans0']['pred1'],data['q0']['ans1']['pred1']] = self.get_prep(context1, q0, [m_name, f_name],
                                                                                     system, system_content)
            [data['q1']['ans0']['pred1'],data['q1']['ans1']['pred1']] = self.get_prep(context1, q1, [m_name, f_name],
                                                                                     system, system_content)
            jsonl_write(output,data)

    def evaluate(self, resfile):
        datas = jsonl_read(resfile)
        C_x1_a = {}

        for data in datas:
            x1 = data['q0']['ans0']['text']
            a = data['q0']['question']
            S_x1t12 = data['q0']['ans0']['pred0']
            S_x1t21 = data['q0']['ans0']['pred1']
            S_x1t12_ = data['q1']['ans0']['pred0']
            S_x1t21_ = data['q1']['ans0']['pred1']
            B_x1 = (S_x1t12+S_x1t21-S_x1t12_-S_x1t21_)/2
            S_x2t12 = data['q0']['ans1']['pred0']
            S_x2t21 = data['q0']['ans1']['pred1']
            S_x2t12_ = data['q1']['ans1']['pred0']
            S_x2t21_ = data['q1']['ans1']['pred1']
            B_x2 = (S_x2t12+S_x2t21-S_x2t12_-S_x2t21_)/2
            C = (B_x1-B_x2)/2
            if x1 not in C_x1_a:
                C_x1_a[x1] = {}
            if a not in C_x1_a[x1]:
                C_x1_a[x1][a]=[]
            C_x1_a[x1][a].append(C)

        miu_x1 = []
        for x1 in C_x1_a:
            gama_x1_a_list = []
            for a in C_x1_a[x1]:
                gama_x1_a_list.append(sum(C_x1_a[x1][a])/len(C_x1_a[x1][a]))
            gama_x1_a_list = [abs(x) for x in gama_x1_a_list]
            miu_x1.append(max(gama_x1_a_list))
        u = sum(miu_x1)/len(miu_x1)
        return None, u, datas
