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
    

		

if __name__ == "__main__":
	main()