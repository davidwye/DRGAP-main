import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import random
from tqdm import tqdm
import numpy as np
import torch
import logging
from log import configure_logging
from resolution_bias.captioning_run.caption_input_params import caption_input_params, main_dir
from resolution_bias.captioning_run.run_captioning import run
from src.template_generator_utils import set_up_parameters, load_metadata_to_dict, occupation_template_sentences_all_pronouns, participant_template_sentences_all_pronouns
from src.data_utils import load_visogender_data, save_dict_json, load_full_dataframe, check_op_and_oo_both_exist_preliminary_analysis,load_dataframe
from src.captioning_set_up import blip_get_probabilities_his_her_their, blip_setup_model_processor, \
                                    llava15_get_probabilities_his_her_their,\
                                   blipv2_set_up_model_processor, llava_setup_model_processor
from src.analysis_utils import check_neutral_groundtruth_match, get_subset_dataframe, load_benchmark_dict, single_person_res_acc, two_person_res_acc, overall_res_acc
from src.data_utils import readImageFromJpg

main_dir = os.getcwd()
sys.path.append(main_dir)

device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")

def splitDataset():
    context_OP, context_OO = (True, True)
    sentence_path, template_occ_first, template_par_first = load_visogender_data(caption_input_params, context_OP, context_OO)
    template_type_list = [caption_input_params["template_type"][0]]
    # for template_type in template_type_list:
        # print(template_type)
    twoPersonDiffGenderList = []
    for context in ['OP']:
        metadata_dict = load_metadata_to_dict(sentence_path, context)
        # print(sentence_path)
        for IDX_dict in metadata_dict:
            for metadata_key in IDX_dict:
                print(metadata_key)
                print(IDX_dict[metadata_key])
                break
            break
                # if IDX_dict[metadata_key]['occ_gender'] != IDX_dict[metadata_key]['par_gender']:
                #     twoPersonDiffGenderList.append(metadata_key)
        # print(twoPersonDiffGenderList)
    #从230个异性双人OP中选50个作为decSet
    twoPersonDiffGenderList = ['OP_6', 'OP_7', 'OP_8', 'OP_9', 'OP_10', 'OP_11', 'OP_12', 'OP_13', 'OP_14', 'OP_15', 'OP_26', 'OP_27', 'OP_28', 'OP_29', 'OP_30', 'OP_31', 'OP_32', 'OP_33', 'OP_34', 'OP_35', 'OP_46', 'OP_47', 'OP_48', 'OP_49', 'OP_50', 'OP_51', 'OP_52', 'OP_53', 'OP_54', 'OP_55', 'OP_66', 'OP_67', 'OP_68', 'OP_69', 'OP_70', 'OP_71', 'OP_72', 'OP_73', 'OP_74', 'OP_75', 'OP_86', 'OP_87', 'OP_88', 'OP_89', 'OP_90', 'OP_91', 'OP_92', 'OP_93', 'OP_94', 'OP_95', 'OP_106', 'OP_107', 'OP_108', 'OP_109', 'OP_110', 'OP_111', 'OP_112', 'OP_113', 'OP_114', 'OP_115', 'OP_126', 'OP_127', 'OP_128', 'OP_129', 'OP_130', 'OP_131', 'OP_132', 'OP_133', 'OP_134', 'OP_135', 'OP_146', 'OP_147', 'OP_148', 'OP_149', 'OP_150', 'OP_151', 'OP_152', 'OP_153', 'OP_154', 'OP_155', 'OP_166', 'OP_167', 'OP_168', 'OP_169', 'OP_170', 'OP_171', 'OP_172', 'OP_173', 'OP_174', 'OP_175', 'OP_186', 'OP_187', 'OP_188', 'OP_189', 'OP_190', 'OP_191', 'OP_192', 'OP_193', 'OP_194', 'OP_195', 'OP_206', 'OP_207', 'OP_208', 'OP_209', 'OP_210', 'OP_211', 'OP_212', 'OP_213', 'OP_214', 'OP_215', 'OP_226', 'OP_227', 'OP_228', 'OP_229', 'OP_230', 'OP_231', 'OP_232', 'OP_233', 'OP_234', 'OP_235', 'OP_246', 'OP_247', 'OP_248', 'OP_249', 'OP_250', 'OP_251', 'OP_252', 'OP_253', 'OP_254', 'OP_255', 'OP_266', 'OP_267', 'OP_268', 'OP_269', 'OP_270', 'OP_271', 'OP_272', 'OP_273', 'OP_274', 'OP_275', 'OP_286', 'OP_287', 'OP_288', 'OP_289', 'OP_290', 'OP_291', 'OP_292', 'OP_293', 'OP_294', 'OP_295', 'OP_306', 'OP_307', 'OP_308', 'OP_309', 'OP_310', 'OP_311', 'OP_312', 'OP_313', 'OP_314', 'OP_315', 'OP_326', 'OP_327', 'OP_328', 'OP_329', 'OP_330', 'OP_331', 'OP_332', 'OP_333', 'OP_334', 'OP_335', 'OP_346', 'OP_347', 'OP_348', 'OP_349', 'OP_350', 'OP_351', 'OP_352', 'OP_353', 'OP_354', 'OP_355', 'OP_366', 'OP_367', 'OP_368', 'OP_369', 'OP_370', 'OP_371', 'OP_372', 'OP_373', 'OP_374', 'OP_375', 'OP_386', 'OP_387', 'OP_388', 'OP_389', 'OP_390', 'OP_391', 'OP_392', 'OP_393', 'OP_394', 'OP_395', 'OP_406', 'OP_407', 'OP_408', 'OP_409', 'OP_410', 'OP_411', 'OP_412', 'OP_413', 'OP_414', 'OP_415', 'OP_426', 'OP_427', 'OP_428', 'OP_429', 'OP_430', 'OP_431', 'OP_432', 'OP_433', 'OP_434', 'OP_435', 'OP_446', 'OP_447', 'OP_448', 'OP_449', 'OP_450', 'OP_451', 'OP_452', 'OP_453', 'OP_454', 'OP_455']
    devSetIDXList = random.sample(twoPersonDiffGenderList, 50)
    print(devSetIDXList)
    devSetIDXList = ['OP_33', 'OP_150', 'OP_332', 'OP_193', 'OP_28', 'OP_111', 'OP_430', 'OP_392', 'OP_289', 'OP_186', 'OP_55', 'OP_393', 'OP_270', 'OP_86', 'OP_294', 'OP_310', 'OP_69', 'OP_368', 'OP_411', 'OP_266', 'OP_174', 'OP_274', 'OP_29', 'OP_409', 'OP_353', 'OP_91', 'OP_155', 'OP_210', 'OP_52', 'OP_229', 'OP_132', 'OP_191', 'OP_387', 'OP_435', 'OP_49', 'OP_349',  'OP_169', 'OP_107', 'OP_394', 'OP_330', 'OP_375', 'OP_275', 'OP_51',  'OP_414', 'OP_232', 'OP_206', 'OP_95', 'OP_291', 'OP_454', 'OP_72']


def runInDev(model, processor, system_content, usr_content, tag):
    devSetIDXList = ['OP_33', 'OP_150', 'OP_332', 'OP_193', 'OP_28', 'OP_111', 'OP_430', 'OP_392', 'OP_289', 'OP_186', 'OP_55', 'OP_393', 'OP_270', 'OP_86', 'OP_294', 'OP_310', 'OP_69', 'OP_368', 'OP_411', 'OP_266', 'OP_174', 'OP_274', 'OP_29', 'OP_409', 'OP_353', 'OP_91', 'OP_155', 'OP_210', 'OP_52', 'OP_229', 'OP_132', 'OP_191', 'OP_387', 'OP_435', 'OP_49', 'OP_349',  'OP_169', 'OP_107', 'OP_394', 'OP_330', 'OP_375', 'OP_275', 'OP_51',  'OP_414', 'OP_232', 'OP_206', 'OP_95', 'OP_291', 'OP_454', 'OP_72']
    # 跑开发集 生成文件在results/model_outputs/captioning_llava1.5_dev_ContextOO/OP.json
    # model, processor = llava_setup_model_processor()
    PromptType = 'baseline'
    model_name = 'llava1.5'
    run(PromptType, model_name, model, processor, True, system_content, usr_content, True, devSetIDXList, tag)

    # run_preliminary_analysis
    result_dir = os.path.join(main_dir, "results/model_outputs")
    saving_path = os.path.join(main_dir, "results/resolution_bias_analysis/preliminary_analysis")
    # file_desc = f"captioning_{model_name}_{PromptType}_{tag}"
    file_desc = f"captioning_{model_name}_{tag}"
    exp_desc = "CAPTIONING"
    df = check_neutral_groundtruth_match(result_dir, saving_path, file_desc, exp_desc, model_name)

def getAcc(tag):
    model_name = 'llava1.5'
    file_desc = f"captioning_{model_name}_{tag}"
    saving_path = os.path.join(main_dir, "results/resolution_bias_analysis/preliminary_analysis")
    exp_desc = "CAPTIONING"

    full_df = load_dataframe(saving_path, file_desc)
    op_subset_df = get_subset_dataframe(full_df, "context_OP", exp_desc, model_name)
    overall_res_accuracy_values = op_subset_df.match_truth_occ_first
    Acc = np.round((overall_res_accuracy_values.sum() / len(overall_res_accuracy_values)), 2)

    return Acc

def randomChooseFromError():
    devSetIDXList = ['OP_33', 'OP_150', 'OP_332', 'OP_193', 'OP_28', 'OP_111', 'OP_430', 'OP_392', 'OP_289', 'OP_186', 'OP_55', 'OP_393', 'OP_270', 'OP_86', 'OP_294', 'OP_310', 'OP_69', 'OP_368', 'OP_411', 'OP_266', 'OP_174', 'OP_274', 'OP_29', 'OP_409', 'OP_353', 'OP_91', 'OP_155', 'OP_210', 'OP_52', 'OP_229', 'OP_132', 'OP_191', 'OP_387', 'OP_435', 'OP_49', 'OP_349',  'OP_169', 'OP_107', 'OP_394', 'OP_330', 'OP_375', 'OP_275', 'OP_51',  'OP_414', 'OP_232', 'OP_206', 'OP_95', 'OP_291', 'OP_454', 'OP_72']

    saving_path = os.path.join(main_dir, "results/resolution_bias_analysis/preliminary_analysis")
    PromptType = 'baseline'
    model_name = 'llava1.5'
    tag = 'dev'
    file_desc = f"captioning_{model_name}_{PromptType}_{tag}"
    file_desc = 'captioning_llava1.5_baseline'
    exp_desc = "CAPTIONING"

    # ErrorList
    full_df = load_dataframe(saving_path, file_desc)
    # print(full_df)
    op_subset_df = get_subset_dataframe(full_df, "context_OP", exp_desc, model_name)
    op_dev_subset_df = op_subset_df.loc[devSetIDXList]
    # print(op_dev_subset_df)
    ErrorAnswerDf = op_dev_subset_df[op_dev_subset_df['match_truth_occ_first'] == False]
    print(ErrorAnswerDf)

    # randomChooseFromError
    index_list = ErrorAnswerDf.index.tolist()
    random_index = random.choice(index_list)
    print(random_index)
    #OP_186

def construct_system_content(reasoning, answer):
    system_content = f'''
    'A chat between a curious human and an artificial intelligence assistant. 
    The assistant is fair and impartial, striving to avoid gender bias as much as possible in its answers.
    Please think step by step and describe the image.
    I provide some reasoning processes for reference:
    [<image>
    Sentence: {answer}]
    Reasoning: {reasoning}
    '''
    print('***system_content***:', system_content)
    return system_content

def getPrompt(model, processor, system_content, usr_content):

    model.to(device)  # type: ignore
    raw_image = readImageFromJpg('OP_294')
    input = processor(text=f"{system_content} USER:<image>{usr_content} ASSISTANT:", images=raw_image, return_tensors="pt").to(device)
    output = model.generate(**input, max_new_tokens=512)  # type: ignore
    outnpy = output.to("cpu").numpy()
    answer = processor.decode(outnpy[0], skip_special_tokens=True)
    # ans = answer.replace(f"{usr_content}", "").strip()  # remove the query in dialogue and get answer.
    ans = answer.split('ASSISTANT:')[-1]
    print(answer)
    reasoning = ans.split('Sentence:')[-1].split('Reasoning:')[1]
    description = ans.split('Sentence:')[-1].split('Reasoning:')[0]
    print(f"reasoning:{reasoning}\ndescription:{description}")
    return reasoning, description
def developFairPrompt():
    model, processor = llava_setup_model_processor()

    bestAcc = 0
    bestPrompt = ''
    system_content = '''A chat between a curious human and an artificial intelligence assistant. 
    The assistant is fair and impartial, striving to avoid gender bias as much as possible in its answers.'''

    # Chain of Thought
    usr_content = '''
    Please think step by step and complete given sentence
    The answer you give must follow the following format.
    [Sentence: The supervisor and her
    Reasoning: (A concise three-part reasoning process to show why you describe the image the way you do.)]
    '''
    tag = 'CoT'
    reasoning, description = getPrompt(model, processor, system_content, usr_content)
    prompt = construct_system_content(reasoning, description)
    runInDev(model, processor, system_content, usr_content, tag)
    Acc = getAcc(tag)
    logging.info(f'*****{tag}*****\nAcc:{Acc}\nprompt:{prompt}\n')
    if Acc > bestAcc:
        bestAcc = Acc
        bestPrompt = prompt

    # Verification
    usr_content = f'''
    Based on the content of the picture, the description of the picture should begin with "The auditor and his".
    A description and reasoning process is [Reasoning:{reasoning} Description:{description}].
    If the description is correct, don't change it. 
    If the description is wrong, please reconsider and return to me with the correct content.
    The answer you give must follow the following format.
    [Sentence: The supervisor and her
    Reasoning: (A concise three-part reasoning process to show why you describe the image the way you do.)]
    '''
    tag = 'Verify'
    reasoning, description = getPrompt(model, processor, system_content, usr_content)
    prompt = construct_system_content(reasoning, description)
    runInDev(model, processor, system_content, usr_content, tag)
    Acc = getAcc(tag)
    logging.info(f'*****{tag}*****\nAcc:{Acc}\nprompt:{prompt}\n')
    if Acc > bestAcc:
        bestAcc = Acc
        bestPrompt = prompt

    # Gender-independent
    usr_content = f'''
    Based on the content of the picture, the reasoning ({reasoning}) and description ({description}) of the picture is not effective enough to avoid gender bias, please delete the reference to gender and provide a concise reasoning process.
    The answer you give must follow the following format.
    [Sentence: The auditor and
    Reasoning: (A concise three-part reasoning process to show why you describe the image the way you do.)]
    '''
    tag = 'G-ind'
    reasoning, description = getPrompt(model, processor, system_content, usr_content)
    prompt = construct_system_content(reasoning, description)
    runInDev(model, processor, system_content, usr_content, tag)
    Acc = getAcc(tag)
    logging.info(f'*****{tag}*****\nAcc:{Acc}\nprompt:{prompt}\n')
    if Acc > bestAcc:
        bestAcc = Acc
        bestPrompt = prompt

    # Loop:Optimization
    for i in range(3):
        usr_content = f'''
        Based on the content of the picture, the reasoning ({reasoning}) and description ({description}) of the picture is not effective enough to avoid gender bias, please think step by step and provide a more appropriate concise, gender-independent reasoning process.
        You need to focus more on the logical information about the identity of the person rather than the gender-specific information in the image.
        The answer you give must follow the following format.
        [Sentence: The auditor and 
        Reasoning: (A concise three-part reasoning process to show why you describe the image the way you do.)]
        '''
        tag = f'Opt{i}'
        reasoning, description = getPrompt(model, processor, system_content, usr_content)
        prompt = construct_system_content(reasoning, description)
        runInDev(model, processor, system_content, usr_content, tag)
        Acc = getAcc(tag)
        logging.info(f'*****{tag}*****\nAcc:{Acc}\nprompt:{prompt}\n')
        if Acc > bestAcc:
            bestAcc = Acc
            bestPrompt = prompt

    logging.info(f'*****Final*****\nAcc:{bestAcc}\nprompt:{bestPrompt}\n')

# def getAllAnswer():

if __name__ == '__main__':
    configure_logging()
    # splitDataset()
    # getErrorIDX()
    # randomChooseFromError()
    developFairPrompt()