import os
import torch
import numpy as np
import argparse
import time
from mp_utils import choices, format_example, gen_prompt, softmax, run_eval
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


def is_eval_success(args) -> bool:
    """judege if eval task is success by checking the result dir"""
    subjects = sorted(
        [f.split(".csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test/"))]
    )
    abs_save_dir = f"{args.save_dir}_{args.num_few_shot}_shot"
    if not os.path.exists(abs_save_dir):
        return False
    for subject in subjects:
        out_file = os.path.join(abs_save_dir, f"results_{subject}.csv")
        if not os.path.exists(out_file):
            # If any result file NOT exist, the eval isn't finished
            return False
    return True


def init_model(args):
    """Initialize models"""
    torch.cuda.empty_cache()
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = args.device
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    model.generation_config = GenerationConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    print(f'model device:{model.device}')
    return model


def eval(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot):
    choice_ids = [tokenizer(choice)["input_ids"][0] for choice in choices]
    cors = []
    all_conf = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        prompt = gen_prompt(
            dev_df=dev_df,
            subject=subject,
            prompt_end=prompt_end,
            num_few_shot=num_few_shot,
            tokenizer=tokenizer,
            max_length=max_length,
            cot=cot,
        )
        label = test_df.iloc[i, test_df.shape[1] - 1]

        with torch.no_grad():
            input_ids = tokenizer([prompt], padding=False)["input_ids"]
            input_ids = torch.tensor(input_ids, device=model.device)
            logits = model(input_ids)["logits"]
            last_token_logits = logits[:, -1, :]
            if last_token_logits.dtype in {torch.bfloat16, torch.float16}:
                last_token_logits = last_token_logits.to(dtype=torch.float32)
            choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
            conf = softmax(choice_logits[0])[choices.index(label)]
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choice_logits[0])]

        all_preds += pred
        all_conf.append(conf)
        cors.append(pred == label)

    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return acc, all_preds, None


def eval_instruct(
    model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot
):
    """eval Qwen/Qwen2-72B-Instruct
    ref: https://huggingface.co/Qwen/Qwen2-72B-Instruct#quickstart
    """
    cors = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]
    records = []
    detail_records = []

    for i in tqdm(range(test_df.shape[0])):
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        prompt = gen_prompt(
            dev_df=dev_df,
            subject=subject,
            prompt_end=prompt_end,
            num_few_shot=num_few_shot,
            tokenizer=tokenizer,
            max_length=max_length,
            cot=cot,
        )
        label = test_df.iloc[i, test_df.shape[1] - 1]

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        start_time = time.time()

        #torch.cuda.synchronize()

        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        total_time = time.time() - start_time

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        #print(f'generated_ids[0]:{generated_ids[0]}')
        detail_records.append(f'{len(generated_ids[0])}/{total_time}')
        records.append(len(generated_ids[0]) / total_time)
        pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #print(f'pred:{pred} len:{len(pred)}')

        # pred, history = model.chat(tokenizer, prompt, history=None)

        if pred and pred[0] in choices:
            cors.append(pred[0] == label)
        all_preds.append(pred.replace("\n", ""))
        if i > int(args.loopcnt):
            break
    print(f'records:{records}')
    print(f'detail_records:{detail_records}')
    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    print(
        "{} results, {} inappropriate formated answers.".format(
            len(cors), len(all_preds) - len(cors)
        )
    )

    mean_speed = np.array(records).mean()
    return acc, all_preds, None, mean_speed

subcategories = {
    "agronomy": ["other"],
    "anatomy": ["biology"],
    "ancient_chinese": ["linguistics", "china specific"],
    "arts": ["arts"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "chinese_civil_service_exam": ["politics", "china specific"],
    "chinese_driving_rule": ["other", "china specific"],
    "chinese_food_culture": ["culture", "china specific"],
    "chinese_foreign_policy": ["politics", "china specific"],
    "chinese_history": ["history", "china specific"],
    "chinese_literature": ["literature", "china specific"],
    "chinese_teacher_qualification": ["education", "china specific"],
    "college_actuarial_science": ["math"],
    "college_education": ["education"],
    "college_engineering_hydrology": ["engineering"],
    "college_law": ["law"],
    "college_mathematics": ["math"],
    "college_medical_statistics": ["statistics"],
    "clinical_knowledge": ["other"],
    "college_medicine": ["other"],
    "computer_science": ["computer science"],
    "computer_security": ["other"],
    "conceptual_physics": ["physics"],
    "construction_project_management": ["other", "china specific"],
    "economics": ["economics"],
    "education": ["education"],
    "elementary_chinese": ["linguistics", "china specific"],
    "elementary_commonsense": ["other", "china specific"],
    "elementary_information_and_technology": ["other"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "ethnology": ["culture", "china specific"],
    "food_science": ["other"],
    "genetics": ["biology"],
    "global_facts": ["global"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_geography": ["geography"],
    "high_school_mathematics": ["math"],
    "high_school_physics": ["physics"],
    "high_school_politics": ["politics", "china specific"],
    "human_sexuality": ["other"],
    "international_law": ["law"],
    "journalism": ["sociology"],
    "jurisprudence": ["law"],
    "legal_and_moral_basis": ["other"],
    "logical": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "marxist_theory": ["philosophy"],
    "modern_chinese": ["linguistics", "china specific"],
    "nutrition": ["other"],
    "philosophy": ["philosophy"],
    "professional_accounting": ["business"],
    "professional_law": ["law"],
    "professional_medicine": ["other"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_study": ["politics"],
    "sociology": ["culture"],
    "sports_science": ["other"],
    "traditional_chinese_medicine": ["other", "china specific"],
    "virology": ["biology"],
    "world_history": ["history"],
    "world_religions": ["global"],
}

categories = {
    "STEM": [
        "physics",

        "computer science",
        "math",

    ],
    "Humanities": ["history", "philosophy", "literature"],
    "Social Science": [
        "business",


        "psychology",

    ],
    "China specific": ["china specific"],
}

TASK_NAME_MAPPING = defaultdict(list)
for k, v in categories.items():
    for subject, subcat in subcategories.items():
        for c in subcat:
            if c in v:
                TASK_NAME_MAPPING[k].append(subject)
total = []
for k,v in TASK_NAME_MAPPING.items():
    for item in v:
        total.append(item)
total = list(set(total))
total = [
    "chinese_foreign_policy",
    "college_law",
    "food_science",
    "sociology",
    "chinese_driving_rule",
    "professional_accounting",
    "chinese_civil_service_exam",
    "ethnology",
    "high_school_geography",
    "machine_learning"
]
total = ['sociology']
print(f'len  total:{len(total)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="/root/autodl-fs/Qwen2-7B-Instruct")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="../results/Qwen2-7B-Chat")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--filter", default=False, help="do filter or not", action='store_true')
    parser.add_argument("--loopcnt", type=int, default=1)

    args = parser.parse_args()
    print(f'args.device:{args.device}')
    print(f'args.filter:{args.filter}')
    print(f'args.loopcnt:{args.loopcnt}')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if is_eval_success(args):
        # eval finished, no need load model anymore, just show the result
        model = None
    else:
        model = init_model(args)

    if "instruct" in args.model_name_or_path.lower():
        run_eval(model, tokenizer, eval_instruct, total, args)
    else:
        run_eval(model, tokenizer, eval, args)
