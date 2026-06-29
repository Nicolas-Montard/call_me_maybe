from . import LlmHandler
from . import ParseArgv
import sys
import json


def main() -> None:
    try:
        argv = ParseArgv.parse_all_args()
        llm_handler: LlmHandler = LlmHandler(
            fn_def_path=argv.functions_definition, prompt_path=argv.input)
        answers = llm_handler.get_all_answers()
        with open(argv.output, "w") as file:
            file.write(json.dumps(answers, indent=4))
    except Exception as e:
        print(e)
        sys.exit()


if __name__ == "__main__":
    main()
