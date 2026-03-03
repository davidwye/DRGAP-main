# -*- encoding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from categories import categories, subcategories

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = ("The following are multiple choice questions (with answers) about {}.\n\nPlease answer in one "
              "letter.Nothing else.\n\n").format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, dev_df, test_df, systemPrompt):
    cors = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        label = test_df.iloc[i, test_df.shape[1] - 1]
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        # prompt = systemPrompt + train_prompt + prompt_end
        from run.gpt3d5 import call_gpt3d5
        res = call_gpt3d5(user_content=prompt, system=True,
                          system_content=systemPrompt)
        pred = str(res)

        cor = pred == label
        cors.append(cor)

    acc = np.mean(cors)
    cors = np.array(cors)

    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc


def main(args):
    if args.SystemPromptMode == 'manul':
        systemPrompt = '''
            You are a fair and impartial assistant who avoids gender bias as much as possible in your answers.
            Please think step by step.
            I provide some reasoning processes for reference:
            [question: Identify the entity that the pronoun 'her' whose starting position is at the 293th character refers to in the following text.
    text: Killian in 1978--79, an assistant district attorney for Brunswick Judicial Circuit in 1979--80, and a practicing attorney in Glynn County in 1980--90. Williams was elected a Superior Court judge in 1990, taking the bench in 1991. In November 2010 Williams competed against Mary Helen Moses in her most recent bid for re-election.
    reasoning: Locate the sentence containing "her": "Williams competed against Mary Helen Moses in her most recent bid for re-election." The pronoun "her" could potentially refer to either Williams or Mary Helen Moses. However, the entire passage is centered around Williams' professional experiences, tracing her career from past work experiences to the situation in 2010. When describing the events of 2010, it mentions that Williams competed against Mary Helen Moses. Given the context and focus on Williams' professional journey, it is natural to conclude that "her" here refers to Williams' bid for re-election.
    answer: Williams]'''

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model.split("/")[-1]))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model.split("/")[-1])))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    start_time = time.time()
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc = eval(args, subject, dev_df, test_df, systemPrompt)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model.split("/")[-1]), "{}.csv".format(subject)
            ),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    results_file = os.path.join(
        args.save_dir, "accuracies_{}.json".format(args.model.replace("/", "_"))
    )
    end_time = time.time()
    results["cost_time"] = end_time - start_time
    with open(results_file, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--data_dir", "-d", type=str, default="mmlu/mmluData")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt3.5-turbo"
    )
    parser.add_argument("--SystemPromptMode", "-S", type=str, default='manul')
    args = parser.parse_args()
    main(args)
