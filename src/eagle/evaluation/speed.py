import json
from transformers import AutoTokenizer
import numpy as np
from evaluation.qwen2_test.load_datasets import load_summary_datasets, rouge_compute_tokenizer

tokenizer=AutoTokenizer.from_pretrained("/root/autodl-fs/Qwen2-7B-Instruct")
jsonl_file = "./qwen2_test/mt_bench/ess-qwen2-7b-fp16-temperature-0.0-summary.jsonl"
jsonl_file_base = "./qwen2_test/mt_bench/ess-qwen2-7b-fp16-temperature-1.0-summary.jsonl"

#jsonl_file = "./qwen2_test/mt_bench/ess-qwen-2-chat-7b-fp16-temperature-0.0.jsonl"
#jsonl_file_base = "./qwen2_test/mt_bench/ess-qwen-2-chat-7b-fp16-baseline-temperature-1.0.jsonl"

#jsonl_file = "./qwen2_test/mt_bench/2.3x-ess-qwen-2-7b-bfp16-temperature-0.0.jsonl"
#jsonl_file_base = "./qwen2_test/mt_bench/2.3x-ess-qwen-2-7b-bfp16-baseline-temperature-1.0.jsonl"
data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)



speeds=[]
pred_answer_list = []
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    pred_answer_list.append(answer[0])
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds.append(tokens/times)


data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)


total_time=0
total_token=0
speeds0=[]
base_answer_list = []

for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    base_answer_list.append(answer[0])
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens


print(f'speed：{speeds}')
print(f'speed0：{speeds0}')

print('speed',np.array(speeds).mean())
print('speed0',np.array(speeds0).mean())
print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())

#print(f'base_answer_list:{base_answer_list}')
#print(f'pred_answer_list:{pred_answer_list}')
rouge_1_fmeasure = rouge_2_fmeasure = rouge_L_fmeasure = 0
for index in range(len(base_answer_list)):
    ret = rouge_compute_tokenizer(pred_answer_list[index], base_answer_list[index], tokenizer)
    print(f'comparing \n {base_answer_list[index]} \n {"=="*20} \n {pred_answer_list[index]}, \n ret:{ret}')
    rouge_1_fmeasure += ret['rouge1'].fmeasure
    rouge_2_fmeasure += ret['rouge2'].fmeasure
    rouge_L_fmeasure += ret['rougeL'].fmeasure
rouge_1_fmeasure /= len(base_answer_list)
rouge_2_fmeasure /= len(base_answer_list)
rouge_L_fmeasure /= len(base_answer_list)
print(f'rouge_1_fmeasure:{rouge_1_fmeasure}')
print(f'rouge_2_fmeasure:{rouge_2_fmeasure}')
print(f'rouge_L_fmeasure:{rouge_L_fmeasure}')




