import csv
import json
import argparse
import os
import torch
import random
import transformers
import time
import re
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset
from peft import (
    PeftModel,
    LoraConfig,
    set_peft_model_state_dict,
)

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 1024  # You need to scale this if you change the model or ntrain value


def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def load_model_with_lora(model_path, device_map, lora_config_path='', lora_weights_path=''):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                              device_map=device_map,
                                                              torch_dtype="auto"
                                                              )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path,
                                                           use_fast=True,
                                                           padding_side="right")
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.model_max_length = max_model_length
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if lora_config_path and lora_weights_path:
        config = LoraConfig.from_pretrained(lora_config_path)
        lora_weights = torch.load(lora_weights_path)
        model = PeftModel(model, config)
        set_peft_model_state_dict(model, lora_weights)
    return model, tokenizer


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    current_path = os.path.dirname(__file__)
    with open(f"{current_path}/cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(model, tokenizer, inference_batch,batch_size=3):
    outputs=[]
    for i in range(0, len(inference_batch), batch_size):
        batch_data = inference_batch[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(batch_data, return_tensors="pt",padding=True,truncation=True).to(model.device)
        outputs.extend(model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True))  
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch

def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
                # print("random hit.")
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(n_train, subject, model, tokenizer, val_df, test_df, output_path, device):
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []

    for i in tqdm(range(len(test_df))):
        k = n_train
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)
            inputs = tokenizer(prompt, return_tensors="pt")
            if device == 'cuda':
                inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)
    pred_batch, response_batch = batch_inference(model, tokenizer, inference_batches)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong


def main():
    if torch.cuda.is_available():
        device_map = "cuda"
    else:
        device_map = "auto"

    model, tokenizer = load_model_with_lora(args.model, device_map, args.lora_config_path, args.lora_weights_path)

    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    full_test_df, full_val_df = load_mmlu_pro()
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)
    logging.info("selected subjects:\n" + "\n".join(selected_subjects))
    print("selected subjects:\n" + "\n".join(selected_subjects))
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))
        acc, corr_count, wrong_count = eval_cot(args.ntrain, subject, model, tokenizer, val_df, test_df, output_path, device_map)
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)


def eval_model(model_path, lora_config_path='', lora_weights_path='', n_train=5, save_dir="model_eval_results"):
    # log file prepare
    os.makedirs(save_dir, exist_ok=True)
    save_result_dir = os.path.join(
        save_dir, "/".join([model_path.split("/")[-1], "CoT", "all"])
    )
    file_prefix = "-".join([model_path.split("/")[-1], "CoT", "all"])
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(save_dir, "summary", file_name)
    os.makedirs(os.path.join(save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    # prepare model, tokenizer and dataset
    if torch.cuda.is_available():
        device_map = "cuda"
    else:
        device_map = "auto"
    model, tokenizer = load_model_with_lora(model_path, device_map, lora_config_path, lora_weights_path)

    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    full_test_df, full_val_df = load_mmlu_pro()
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    selected_subjects = all_subjects

    logging.info("selected subjects:\n" + "\n".join(selected_subjects))

    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))
        acc, corr_count, wrong_count = eval_cot(n_train, subject, model, tokenizer, val_df, test_df, output_path, device_map)
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    return sta_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=1, help="few-shot examples amount, default is 3")
    parser.add_argument("--selected_subjects", "-sub", type=str, default="history", help="selected subjects: biology, business, chemistry, computer science, economics, engineering, health, history, law, math, philosophy, physics, psychology, other, all. default is 'all'")
    parser.add_argument("--save_dir", "-s", type=str, default="eval_results", help="evaluation results save dir, default is 'eval_results'")
    parser.add_argument("--lora_config_path", "-lc",type=str, help="lora config folder path")
    parser.add_argument("--lora_weights_path", "-lw",type=str, help="lora weights bin file path")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv", help="global log record file, default is 'eval_record_collection.csv'")
    parser.add_argument("--model", "-m", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct", help="local model path")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    main()

