try:
    from pydantic import BaseModel, Field, model_validator, ValidationError, ConfigDict
except ModuleNotFoundError as e:
    print("pydantic is not installed on this system")
    print(e)
    exit()
from . import JsonManager
import json
from llm_sdk import Small_LLM_Model
import numpy as np

class LlmHandler(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vocab_path: dict|None = Field(default=None)
    llm: Small_LLM_Model = Small_LLM_Model()
    fn_names: list[str]|None = Field(default=None)
    fn_descriptions: list[str]|None = Field(default=None)
    token_to_id: dict = {}
    id_to_token: dict = {}

    @model_validator(mode='after')
    def init_values(self):
        vocab_path = self.llm.get_path_to_vocab_file()
        with open(vocab_path) as f: 
            self.token_to_id = json.load(f)
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.fn_names = [x['name'] for x in JsonManager.fn_def]
        self.fn_descriptions = [x['description'] for x in JsonManager.fn_def]
        return self
    
    def get_answer(self, user_prompt: str):
        pass

    def get_valid_input_name(self, answer: list[int], logits: list[float]) \
        -> list[float]:
        if (self.fn_names is None or self.fn_descriptions is None):
            raise ValueError("fn_names or fn_descriptions are none")
        for i, logit in enumerate(logits):
            temp_answer = answer.copy()
            temp_answer.append(i)
            possible_str = self.llm.decode(temp_answer).strip()
            if not any(possible_str in fn_name for fn_name in self.fn_names):
                logits[i] = float('-inf')
        return logits

    def get_name_of_func(self, user_prompt: str) -> str:
        if (self.fn_names is None or self.fn_descriptions is None):
            raise ValueError("fn_names or fn_descriptions are none")
        prompt = self.get_prompt_for_name(user_prompt)
        data_list: list[int] = self.llm.encode(prompt)[0].tolist()
        answer: list[int] = []
        while(True):
            logits = self.llm.get_logits_from_input_ids(data_list + answer)
            next_char_id = int(np.argmax(self.get_valid_input_name(answer, logits)))
            answer.append(next_char_id)
            result = self.llm.decode(answer).strip()
            if (result in self.fn_names):
                break
        return result
    
    def get_prompt_for_name(self, user_prompt: str) -> str:
        fn_list = ""
        for fn in JsonManager.fn_def:
            fn_list += f"- {fn['name']}: {fn['description']}\n"
        system = (
        "You are a function selection assistant. "
        "Given a user request, respond with ONLY the function name. "
        "No explanation, no punctuation, just the function name. /no_think\n\n"
        f"Available functions:\n{fn_list}"
        )

        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    def get_prompt_for_args(self, user_prompt: str, fn_def: dict) -> str:
        params = ", ".join(
            f"{name}: {p['type']}"
            for name, p in fn_def['parameters'].items()
        )

        system = (
            f"You are a function calling assistant.\n"
            f"The user wants to call: {fn_def['name']}({params})\n"
            f"Extract the arguments from the user request.\n"
            f"Respond with ONLY a valid JSON object. /no_think"
        )

        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )