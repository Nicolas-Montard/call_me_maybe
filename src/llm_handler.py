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
from typing import Any

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
    
    def get_answer_for_one_function(self):
        pass
    
    def get_all_args_of_func(self, user_prompt: str, fn_def: dict):
        answers = []
        for i in range(len(fn_def['parameters'])):
            answers.append(
                self.get_valid_input_arg(fn_def, i + 1, user_prompt))
        return answers
    
    def validate_string(self, answer: list[int], logits: list[float]) -> list[float]:
        if (len(answer) == 0):
            for i, logit in enumerate(logits):
                if(self.llm.decode([i]) != "\""):
                    logits[i] = float('-inf')
        return logits

    def get_valid_string(self, prompt: str) -> str:
        data_list: list[int] = self.llm.encode(prompt)[0].tolist()
        answer: list[int] = []
        while(len(answer) < 2 or self.id_to_token[answer[-1]] != "\""):
            logits = self.llm.get_logits_from_input_ids(data_list + answer)
            next_char_id = int(np.argmax(self.validate_string(answer, logits)))
            answer.append(next_char_id)
        return self.llm.decode(answer)
    

    def get_valid_input_arg(self, function: dict, arg_nb: int,
                            user_prompt: str) -> Any:
        arg_name, arg_type = list(function["parameters"].items()[arg_nb - 1])
        result = None
        if (arg_type == "string"):
            result = self.get_valid_string(
                self.get_prompt_for_single_arg(user_prompt, function, arg_name, arg_type))
        elif (arg_type == "number"):
            result = self.get_valid_number(self.get_prompt_for_single_arg(
                user_prompt, function, arg_name, arg_type
                ))
        elif (arg_type == "boolean"):
            result = self.get_valid_boolean(self.get_prompt_for_single_arg(
                user_prompt, function, arg_name, arg_type
                ))
        elif (arg_type == "array"):
            result = self.get_valid_array(self.get_prompt_for_single_arg(
                user_prompt, function, arg_name, arg_type
                ))
        else:
            result = self.get_valid_object(self.get_prompt_for_single_arg(
                user_prompt, function, arg_name, arg_type
            ))
        return result
    
    def validate_number(self, answer: list[int], logits: list[float]) -> list[float]:
        for i in range(len(logits)):
            current = self.llm.decode(answer + [i]).strip('"')
            if len(answer) == 0:
                if self.id_to_token.get(i, "") != "\"":
                    logits[i] = float('-inf')
                continue
            if self.id_to_token.get(i, "") == "\"":
                continue
            try:
                float(current)
            except ValueError:
                logits[i] = float('-inf')
        return logits

    def get_valid_number(self, prompt: str) -> float:
        data_list: list[int] = self.llm.encode(prompt)[0].tolist()
        answer: list[int] = []
        while len(answer) < 2 or self.id_to_token[answer[-1]] != "\"":
            logits = self.llm.get_logits_from_input_ids(data_list + answer)
            next_char_id = int(np.argmax(self.validate_number(answer, logits)))
            answer.append(next_char_id)
        result = self.llm.decode(answer).strip('"')
        return float(result)
    
    def validate_boolean(self, answer: list[int], logits: list[float]) -> list[float]:
        for i in range(len(logits)):
            temp = self.llm.decode(answer + [i]).strip()
            if not ("true".startswith(temp) or "false".startswith(temp)):
                logits[i] = float('-inf')
        return logits

    def get_valid_boolean(self, prompt: str) -> bool:
        data_list: list[int] = self.llm.encode(prompt)[0].tolist()
        answer: list[int] = []
        while True:
            logits = self.llm.get_logits_from_input_ids(data_list + answer)
            logits = self.validate_boolean(answer, logits)
            next_char_id = int(np.argmax(logits))
            answer.append(next_char_id)
            result = self.llm.decode(answer).strip()
            if result in {"true", "false"}:
                break
        return bool(result)
    
    def validate_array(self, answer: list[int], logits: list[float]) -> list[float]:
        for i in range(len(logits)):
            temp = self.llm.decode(answer + [i]).strip()
            if len(answer) == 0 and not temp.startswith('['):
                logits[i] = float('-inf')
                continue
            try:
                json.loads(temp)
                continue
            except json.JSONDecodeError:
                pass
            try:
                json.loads(temp + ']')
                continue
            except json.JSONDecodeError:
                pass
            try:
                json.loads(temp + '0]')
                continue
            except json.JSONDecodeError:
                pass
            try:
                json.loads(temp + '"]')
                continue
            except json.JSONDecodeError:
                pass
            logits[i] = float('-inf')
        return logits

    def get_valid_array(self, prompt: str) -> list[Any]:
        data_list: list[int] = self.llm.encode(prompt)[0].tolist()
        answer: list[int] = []
        while True:
            logits = self.llm.get_logits_from_input_ids(data_list + answer)
            logits = self.validate_array(answer, logits)
            next_char_id = int(np.argmax(logits))
            answer.append(next_char_id)
            result = self.llm.decode(answer).strip()
            if result.endswith(']'):
                break
        return json.loads(result)
    
    def validate_object(self, answer: list[int], logits: list[float]) -> list[float]:
        for i in range(len(logits)):
            temp = self.llm.decode(answer + [i]).strip()
            if len(answer) == 0 and not temp.startswith('{'):
                logits[i] = float('-inf')
                continue
            try:
                json.loads(temp)
                continue
            except json.JSONDecodeError:
                pass
            try:
                json.loads(temp + '}')
                continue
            except json.JSONDecodeError:
                pass
            try:
                json.loads(temp + '0}')
                continue
            except json.JSONDecodeError:
                pass
            try:
                json.loads(temp + '"}')
                continue
            except json.JSONDecodeError:
                pass
            logits[i] = float('-inf')
        return logits

    def get_valid_object(self, prompt: str) -> dict:
        data_list: list[int] = self.llm.encode(prompt)[0].tolist()
        answer: list[int] = []
        while True:
            logits = self.llm.get_logits_from_input_ids(data_list + answer)
            logits = self.validate_object(answer, logits)
            next_char_id = int(np.argmax(logits))
            answer.append(next_char_id)
            result = self.llm.decode(answer).strip()
            if result.endswith('}'):
                break
        return json.loads(result)

    def get_prompt_for_single_arg(self, user_prompt: str, fn_def: dict, arg_name: str, arg_type: str) -> str:
        type_instructions = {
            "string": 'Respond with ONLY a JSON string (with quotes). Example: "hello"',
            "number": 'Respond with ONLY a JSON number (with quotes). Example: "42.0" or "3.14"',
            "boolean": 'Respond with ONLY a JSON boolean (without quotes). Example: true or false',
            "array": 'Respond with ONLY a JSON array of primitives. Example: [1, 2, 3] or ["a", "b"]',
            "object": 'Respond with ONLY a JSON object with primitive values. Example: {"key": "value", "count": 42}',
        }
        instruction = type_instructions.get(arg_type, f"Respond with ONLY a JSON {arg_type} (with quotes).")

        system = (
            f"You are an argument extraction assistant.\n"
            f"The user wants to call the function: {fn_def['name']}\n"
            f"Function description: {fn_def['description']}\n"
            f"Extract ONLY the value of the argument '{arg_name}' (type: {arg_type}) "
            f"from the user request.\n"
            f"{instruction} /no_think"
        )

        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        