try:
    from pydantic import BaseModel
except ModuleNotFoundError as e:
    print("pydantic is not installed on this system")
    print(e)
    exit()
import json

class JsonManager(BaseModel):
    prompt: dict
    fn_def: dict

    @classmethod
    def load_files(cls):
        with open("data/input/functions_definition.json", "r") as file:
            cls.fn_def = json.load(file)
        with open("data/input/function_calling_tests.json", "r") as file:
            cls.prompt = json.load(file)