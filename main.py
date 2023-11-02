import argparse
import random
import torch
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import build_chat_input, alignment_check
from defaults import model_dirs, test_prefixes
from test_samples import test_samples

def robustness_check(query, tokenizer, model, args):
    # check original response state
    input_string = build_chat_input(query, model)
    input_token = tokenizer([input_string], padding=args.padding, truncation=False, return_tensors='pt').to("cuda")
    input_token_num = input_token['input_ids'].shape[-1]
    alignment_flag = alignment_check(input_token, tokenizer, model, input_token_num, test_prefixes, max_new_tokens=args.max_new_tokens)
    if not alignment_flag:
        print(f"original query is not safe!")
        return False
    print(f"### query ###: {query}")
    print(f"### response ###: {alignment_flag}")
    # Monte Carlo sampling
    query_tokens = tokenizer([query], padding=args.padding, truncation=False, return_tensors='pt').to("cuda")
    query_tokens_num = query_tokens['input_ids'].shape[-1]
    dropping_num = int(args.dropping_rate * query_tokens_num)
    false_num = 0
    for sample in range(args.sampling_num):
        token_indexs_to_remove = random.sample(range(query_tokens_num), dropping_num)
        query_token_ids = query_tokens['input_ids']

        dropped_query_token_ids = [query_token_ids[:, i] for i in range(query_tokens_num) if
                                   i not in token_indexs_to_remove]
        dropped_query_token_ids = torch.cat(dropped_query_token_ids).unsqueeze(0)
        dropped_query_string = tokenizer.batch_decode(dropped_query_token_ids, skip_special_tokens=True)[0]
        # check alignment
        dropped_input_string = build_chat_input(dropped_query_string, model)
        dropped_input_token = tokenizer([dropped_input_string], padding=args.padding, truncation=False, return_tensors='pt').to("cuda")
        dropped_input_token_num = dropped_input_token['input_ids'].shape[-1]
        if not alignment_check(dropped_input_token, tokenizer, model, dropped_input_token_num, test_prefixes, max_new_tokens=args.max_new_tokens):
            false_num += 1

    false_rate =  false_num / args.sampling_num
    if false_rate < args.threshold:
        print(f"### RA-LLM ###: query is safe, false rate: {false_rate}")
    else:
        print(f"### RA-LLM ###: query is not safe, false rate: {false_rate}")

    return alignment_flag



def main(args):
    model = AutoModelForCausalLM.from_pretrained(model_dirs[args.test_model][0], device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dirs[args.test_model][0], trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = 'left'

    for q in range(len(test_samples[args.test_model])):
        print(f"------------------------------- adv case {q} -------------------------------------")
        query = test_samples[args.test_model][q]
        response = robustness_check(query, tokenizer, model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='llm defends')
    parser.add_argument('--sampling_num', type=int, default=20)
    parser.add_argument('--max_new_tokens', type=int, default=10)
    parser.add_argument('--dropping_rate', type=float, default=0.3)
    parser.add_argument('--padding', type=bool, default=True)
    parser.add_argument('--test_model', type=str, default='vicuna')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--manual_seed', default=0, type=int)
    parser.add_argument('--device', default='0', type=str)
    args = parser.parse_args()

    np.random.seed(seed = args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.test_model == "vicuna":
        args.padding = True
    else:
        args.padding = False

    main(args)

