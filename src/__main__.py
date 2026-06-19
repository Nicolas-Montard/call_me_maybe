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
from . import LlmHandler

def main():
    try:
        JsonManager.load_files()
    except (ValidationError, FileNotFoundError, JSONDecodeError) as e:
        print(f"ERROR: {e}")
        exit()
    llm_handler: LlmHandler = LlmHandler()
    answers = llm_handler.get_all_answers()
    print(json.dumps(answers, indent=4))
    


		

if __name__ == "__main__":
	main()