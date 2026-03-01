import json
from reasoners.benchmark import BWEvaluator
import fire
from reasoners.lm.openai_model import OpenAIModel

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

class CoTReasoner():
    def __init__(self, base_model, temperature=0.8, model_type="completion", mode_type="baseline",base_lm=None):
        self.base_model = base_model
        self.temperature = temperature
        self.model_type = model_type
        self.mode_type = mode_type
        self.base_lm = base_lm

    def __call__(self, example, prompt=None):
        if self.mode_type == "io":
            print("io mode")
            inputs = load_prompt("case_study/rule_patterns/blocksworld/prompts/baseline.txt")
        elif self.mode_type == "cot":
            if self.base_lm == "gpt-4o" or "claude" in self.base_lm:
                inputs = load_prompt("case_study/rule_patterns/blocksworld/prompts/baseline_cot_v1.txt")
            else:
                inputs = load_prompt("case_study/rule_patterns/blocksworld/prompts/baseline_cot_v3.txt")
        elif self.mode_type == "ours":
            if self.base_lm == "gpt-4o":
                inputs = load_prompt("case_study/rule_patterns/blocksworld/prompts/ours_v5.txt")
            elif self.base_lm == "qwen-max-0919" or self.base_lm == "gpt-3.5-turbo":
                inputs = load_prompt("case_study/rule_patterns/blocksworld/prompts/ours_v2.txt")
            else:
                inputs = load_prompt("case_study/rule_patterns/blocksworld/prompts/ours_v1.txt")
        inputs = inputs.replace("<init_state>", example["init"]).replace("<goals>", example["goal"])
        if self.model_type == "completion":
            output = self.base_model.generate([inputs],
                                          hide_input=True,
                                          do_sample=True,
                                          temperature=self.temperature,
                                          eos_token_id='\n[').text[0][:-1].strip()
        elif self.model_type == "chat":
            output = self.base_model.generate([inputs],
                                          hide_input=True,
                                          do_sample=True,
                                          temperature=self.temperature).text[0].replace("[PLAN END]", "").strip()            
        return output

def main(base_lm, data_path, prompt_path, disable_log=False, config_file: str = "case_study/rule_patterns/blocksworld/data/bw_config.yaml", domain_file: str = "case_study/rule_patterns/blocksworld/data/generated_domain.pddl", resume=0, log_dir=None, temperature=0.8,mode_type="baseline"):

    base_model = OpenAIModel(base_lm, additional_prompt="CONTINUE")
    with open(prompt_path) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(base_model, temperature=temperature, model_type="chat", mode_type=mode_type,base_lm=base_lm)
    evaluator = BWEvaluator(config_file=config_file, domain_file=domain_file, data_path=data_path, init_prompt=prompt, disable_log=disable_log, output_extractor=lambda x:x, sample_prompt_type="rap") # rap prompt includes cot
    

    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=0, resume=resume, log_dir=log_dir)

    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    fire.Fire(main)