from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from peft.utils import prepare_model_for_kbit_training
from omegaconf import OmegaConf
import torch


def get_model_and_tokenizer(cfg_path):
    config_detail = OmegaConf.load(cfg_path)

    if config_detail.model.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif config_detail.model.torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = None

    if config_detail.model.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_compute_dtype=torch_dtype
                                                 )
    elif config_detail.model.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    if config_detail.model.device_map in ["cuda", "cpu", "mps"]:
        device_map = config_detail.model.device_map
        if device_map == 'mps':
            quantization_config = None
    else:
        if torch.cuda.is_available():
            device_map = "cuda"
        else:
            device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        config_detail.model.model_path,
        quantization_config=quantization_config,
        device_map={"": device_map},
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config_detail.model.model_path, use_fast=True, padding_side="right"
    )

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.model_max_length = 2048
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(
        r=config_detail.model.lora.peft_lora_r,
        lora_alpha=config_detail.model.lora.peft_lora_alpha,
        lora_dropout=0.055,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config_detail.model.target_modules),
    )

    model = get_peft_model(model, peft_config)
    peft_config.save_pretrained(config_detail.sft.training_arguments.output_dir)
    return model, tokenizer
