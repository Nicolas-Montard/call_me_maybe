from json import JSONDecodeError
import json
import numpy as np
try:
    from pydantic import ValidationError
except ModuleNotFoundError as e:
    print("pydantic is not installed on this system")
    print(e)
    exit()
from . import JsonManager
from llm_sdk import Small_LLM_Model
from . import LlmHandler

def main():
    try:
        JsonManager.load_files()
    except (ValidationError, FileNotFoundError, JSONDecodeError) as e:
        print(f"ERROR: {e}")
        exit()
    llm_handler: LlmHandler = LlmHandler()
    # system = (
    #     f"You are an argument extraction assistant.\n"
    #     f"The user wants to call the function: put_in_array\n"
    #     f"Function description: Add two string in an array and return it\n"
    #     f"Extract ONLY the value of the argument 'a' (type: array) "
    #     f"from the user request.\n"
    #     f"Respond with ONLY a JSON array of primitives. /no_think"
    # )
    # prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
    # f"<|im_start|>Put 0 and 1 in a tab\n<|im_end|>\n"
    # f"<|im_start|>assistant\n")
    # answer = llm_handler.llm.encode('[')[0].tolist()
    # prompt_encoded = llm_handler.llm.encode(prompt)[0].tolist()
    # logits = llm_handler.validate_array(answer, llm_handler.llm.get_logits_from_input_ids(prompt_encoded + answer))
    # next_char_id = int(np.argmax(logits))
    # answer.append(next_char_id)
    # print(llm_handler.llm.decode(answer))
    answer = llm_handler.get_answer_for_one_function("put 'test' in an object with the key '1'")
    print(json.dumps(answer, indent=4))


		

if __name__ == "__main__":
	main()