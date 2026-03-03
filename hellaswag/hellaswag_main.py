import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from utils.file_utils import jsonl_read, jsonl_write
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig
from gpt3d5 import call_gpt3d5
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
systemPromptMap =
class HellaSwagLlama():

    def __init__(self, modelname):
        if modelname == 'llama3':
            self.model_name = "meta/Meta-Llama-3-8B-Instruct"
        elif modelname == 'llama2Alpaca':
            self.model_name = "meta/Alpaca_llama-2-naive-7b-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, device_map=DEVICE)
        self.model = LlamaForCausalLM.from_pretrained(self.model_name, device_map=DEVICE)

    def run(self, SystemPromptMode, modelName):
        if SystemPromptMode == 'manul':
            systemPrompt = '''
            You are a fair and impartial assistant who avoids gender bias as much as possible in your answers.
            Please think step by step.
            I provide some reasoning processes for reference:
            [question: Identify the entity that the pronoun 'her' whose starting position is at the 293th character refers to in the following text.
    text: Killian in 1978--79, an assistant district attorney for Brunswick Judicial Circuit in 1979--80, and a practicing attorney in Glynn County in 1980--90. Williams was elected a Superior Court judge in 1990, taking the bench in 1991. In November 2010 Williams competed against Mary Helen Moses in her most recent bid for re-election.
    reasoning: Locate the sentence containing "her": "Williams competed against Mary Helen Moses in her most recent bid for re-election." The pronoun "her" could potentially refer to either Williams or Mary Helen Moses. However, the entire passage is centered around Williams' professional experiences, tracing her career from past work experiences to the situation in 2010. When describing the events of 2010, it mentions that Williams competed against Mary Helen Moses. Given the context and focus on Williams' professional journey, it is natural to conclude that "her" here refers to Williams' bid for re-election.
    answer: Williams]
            '''
        systemPrompt = systemPromptMap[systemPromptMode]

        data = jsonl_read('datasets/hellaswag_val.jsonl')
        for each in data:
            ctx = each['ctx']
            endings = each['endings']
            probs = self.computer_complexity(systemPrompt+ctx, endings)
            print(probs)
            each['probs'] = probs
            jsonl_write(f'hellaswagResult/{modelName}_{SystemPromptMode}_probs.jsonl', each)

    def computer_complexity(self, context, endings):
        probabilities = []

        for ending in endings:
            inputs = self.tokenizer(context + ending, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # 计算每个token的困惑度
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1))
            perplexity = torch.exp(loss)
            probabilities.append(-perplexity.item())  # 负困惑度，越大越好

        temperature = 1.0  # 可调节的温度参数
        probs = F.softmax(torch.tensor(probabilities).to(DEVICE) / temperature, dim=0)

        return probs.cpu().tolist()

def runLlama(modelName):

    for SystemPromptMode in ['manul']:
        Obj = HellaSwagLlama(modelName)
        Obj.run(SystemPromptMode, modelName)
        del Obj
class HellaSwagGPT():

    def __init__(self):
        pass

    def run(self, SystemPromptMode, modelName='gpt3d5'):
        data = jsonl_read('datasets/hellaswag_val.jsonl')
        for each in data:
            ctx = each['ctx']
            endings = each['endings']
            res = self.get_answer(SystemPromptMode,ctx,endings)
            each['res'] = res
            jsonl_write(f'hellaswagResult/{modelName}_{SystemPromptMode}_probs.jsonl', each)


    def get_answer(self, SystemPromptMode, ctx, endings):
        if SystemPromptMode == 'manul':
            systemPrompt = '''
                You are a fair and impartial assistant who avoids gender bias as much as possible in your answers.
                Please think step by step.
                '''

        res = call_gpt3d5(user_content=f"""Given the text: '{ctx}',
                    consider three options: 0.{endings[0]} 1.{endings[1]} 2.{endings[2]} 3.{endings[3]}
                    Which option is the most suitable to follow the given text?
                    Answer in one number, nothing else.""", system=True, system_content=systemPrompt)
        return str(res)

def runGPT():
    HG = HellaSwagGPT()
    modelName = 'gpt3d5'
    for SystemPromptMode in ['manul']:
        HG.run(SystemPromptMode, modelName)


def find_max_index(lst):
    if not lst:
        return None
    max_index = 0

    # 遍历列表，找到最大值的索引
    for i in range(1, len(lst)):
        if lst[i] > lst[max_index]:
            max_index = i

    return max_index

def llama_result_deal(modelName):
    from utils.file_utils import find_files_with_string
    fileList = find_files_with_string('../hellaswagResult', modelName)
    for filepath in fileList:
        if 'manul' in filepath:
            print(filepath)
            data = jsonl_read(filepath)
            for each in data:
                each['res'] = find_max_index(each['probs'])
                jsonl_write(os.path.join('../ResultHellaSwag/', os.path.basename(filepath)), each)

def accuracy(modelName):
    res = []
    # for filename in ['','Fair_','Fair+ aggregate_','Fair+ aggregate QA_', 'Fair+ CDF gpt_']:
    for filename in ['manul']:
        filepath = os.path.join('../ResultHellaSwag',f'{modelName}_{filename}_probs.jsonl')
        print(os.path.basename(filepath))
        data = jsonl_read(filepath)
        totalNum = 0
        correctNum = 0
        for each in data:
            totalNum += 1
            if modelName == 'gpt3d5' and str(each['label']) in each['res']:
                correctNum += 1
            elif each['label'] == each['res']:
                correctNum += 1
        res.append(correctNum/totalNum)
    print(res)

if __name__ == '__main__':
    optName = sys.argv[1]
    if optName == 'llama':
        runLlama('llama3')
        runLlama('llama2Alpaca')
    if optName == 'gpt':
        runGPT()
    # if optName == 'data_deal':
    #     llama_result_deal('llama2Alpaca')
    #     llama_result_deal('llama3')
    #     accuracy('llama3')
    #     accuracy('llama2Alpaca')
    #     accuracy('gpt3d5')
   # accuracy('llama2Align')


