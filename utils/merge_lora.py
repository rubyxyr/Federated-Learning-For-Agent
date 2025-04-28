import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from peft import (
    PeftModel,
    LoraConfig,
    set_peft_model_state_dict,
)

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    load_in_8bit=True,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("model_path")
lora_weights_path = "lora weight"
lora_config_path = "lora config"
config = LoraConfig.from_pretrained(lora_config_path)
lora_weights = torch.load(lora_weights_path)

model = PeftModel(model, config)

set_peft_model_state_dict(model, lora_weights)


print("")
messages = [{"role": "user", "content": "who are you?"}]

input_ids = tokenizer.apply_chat_template(
    conversation=messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
output_ids = model.generate(input_ids.to("cuda"), max_new_tokens=2048)

response = tokenizer.decode(
    output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
)

model.merge_and_unload()
