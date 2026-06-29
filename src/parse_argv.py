import argparse
import os


class ParseArgv():
    @staticmethod
    def valid_input_path(path: str) -> str:
        if not os.path.isfile(path):
            raise argparse.ArgumentTypeError(
                f"The file '{path}' does not exist")
        if not path.endswith(".json"):
            raise argparse.ArgumentTypeError(
                f"The file {path} must be a .json")
        return path

    @staticmethod
    def valid_output_path(path: str) -> str:
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            raise argparse.ArgumentTypeError(
                f"The directory '{parent}' does not exist")
        if not path.endswith(".json"):
            raise argparse.ArgumentTypeError(
                "The output file must be a .json")
        return path

    @staticmethod
    def parse_all_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--functions_definition",
            type=ParseArgv.valid_input_path,
            default="data/input/functions_definition.json",
        )
        parser.add_argument(
            "--input",
            type=ParseArgv.valid_input_path,
            default="data/input/function_calling_tests.json",
        )
        parser.add_argument(
            "--output",
            type=ParseArgv.valid_output_path,
            default="data/output/function_calls.json",
        )
        return parser.parse_args()
