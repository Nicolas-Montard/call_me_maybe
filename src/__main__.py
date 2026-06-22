from json import JSONDecodeError
import json
import sys
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

    with open("data/output/test.json", "w") as file:
        try:
            file.write(json.dumps(answers, indent=4))
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit()
    

if __name__ == "__main__":
	main()