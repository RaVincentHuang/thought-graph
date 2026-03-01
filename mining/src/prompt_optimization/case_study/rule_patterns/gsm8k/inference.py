import json
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.hf_model import HFModel
from reasoners.benchmark import GSM8KEvaluator
import utils
from typing import Literal
import fire
import transformers
import os

def load_prompt(prompt_path):
    with open(prompt_path, 'r',encoding='utf-8') as f:
        return f.read()

class CoTReasoner():
    def __init__(self, base_model, n_sc=1, temperature=0, bs=1, log_dir=None,exp_mode=None):
        assert n_sc == 1 or temperature > 0, \
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs
        self.log_dir = log_dir
        self.exp_mode = exp_mode
    def __call__(self, example, prompt=None):
        if self.exp_mode == "io":
            inputs = load_prompt("case_study/rule_patterns/gsm8k/prompts/io.txt").replace("{QUESTION}", example)
        elif self.exp_mode == "cot":
            inputs = load_prompt("case_study/rule_patterns/gsm8k/prompts/cot.txt").replace("{QUESTION}", example)
        elif self.exp_mode == "ours":
            inputs = load_prompt("case_study/rule_patterns/gsm8k/prompts/ours_v1.txt").replace("{QUESTION}", example)
        outputs = []
        do_sample = True
        if self.temperature == 0 and isinstance(self.base_model, HFModel):
            print("Using greedy decoding with HF model. Set do_sample=False")
            self.temperature == 1.0
            do_sample = False
        if isinstance(self.base_model, OpenAIModel):
            eos_token_id = []
        elif isinstance(self.base_model.model, transformers.GemmaForCausalLM):
            eos_token_id = [108]
        elif isinstance(self.base_model.model, transformers.MistralForCausalLM) or isinstance(self.base_model.model, transformers.MixtralForCausalLM):
            eos_token_id = [13]
        elif self.base_model.model.config.architectures[0] == 'InternLM2ForCausalLM':
            eos_token_id = [364,402,512,756]
        elif self.base_model.model.config.architectures[0] == 'Qwen2ForCausalLM':
            eos_token_id = [198,271,382,624,151645]
        else:
            eos_token_id = ["\n\n", ".\n", "\n", ".\n\n"]
        for i in range((self.n_sc - 1) // self.bs + 1):
            local_bs = min(self.bs, self.n_sc - i * self.bs)
            outputs += self.base_model.generate([inputs] * local_bs,
                                            hide_input=True,
                                            do_sample=do_sample,
                                            temperature=self.temperature,
                                            eos_token_id = eos_token_id).text
        outputs= [o.strip() if o.strip().endswith(".") else o.strip() + "." for o in outputs]
        return outputs
    
def main(base_lm:Literal['llama2',' exllama', 'llama3', 'gpt-4', 'gpt-3.5-turbo', 'glm-4-plus', 'gpt-4o', 'gpt-4o-mini', 'qwen-max-latest','hf', 'glm-4-9b', 'chatglm3-6b', 'llama-3-8b-instruct', 'qwen1.5-7b', 'mistral-7b'],model_dir=None, lora_dir=None, mem_map=None, batch_size=1, prompt="case_study/rule_patterns/gsm8k/prompts/cot.json", resume=0, log_dir=None, temperature=0, n_sc=1, quantized='int8',llama_size=None,exp_mode=None):

    if base_lm == "gpt-3.5-turbo":
        base_model = OpenAIModel("gpt-3.5-turbo", additional_prompt="ANSWER")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir, max_batch_size=1, max_new_tokens=512, quantized=None)
    else:
        raise ValueError(f"Unknown base_lm: {base_lm}")
    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(base_model, temperature=temperature, n_sc=n_sc, bs=batch_size, log_dir=log_dir,exp_mode=exp_mode)
    evaluator = GSM8KEvaluator(
                 output_extractor=utils.cot_sc_extractor,
                 answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt,
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="cot")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=0, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    fire.Fire(main)


