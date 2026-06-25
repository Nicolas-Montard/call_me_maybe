try:
    from pydantic import BaseModel, model_validator
except ModuleNotFoundError as e:
    print("pydantic is not installed on this system")
    print(e)
    exit()
import json
from enum import Enum



class TypeValue(str, Enum):
    string = "string"
    number = "number"
    boolean = "boolean"
    array = "array"
    object = "object"
    integer = "integer"


class Prompt(BaseModel):
    prompt: str

class Type(BaseModel):
    type: TypeValue

class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, Type]
    returns: Type


class JsonManager(BaseModel):
    prompts: list[Prompt]
    fn_def: list[FunctionDef]

    @staticmethod
    def load_files(fn_def_path: str, prompt_path: str):
        with open(fn_def_path, "r") as file:
            fn_def = json.load(file)
        with open(prompt_path, "r") as file:
            prompts = json.load(file)
        return [prompts, fn_def]
    
    def get_prompts(self):
        return [p.prompt for p in self.prompts]
    
    def get_fn_defs(self):
        return [fn.model_dump(mode="json") for fn in self.fn_def]