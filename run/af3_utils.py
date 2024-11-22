import os
import argparse
from typing import Sequence, Dict, Any
import json
import pathlib
import logging
from alphafold3.common.folding_input import Input, check_unique_sanitised_names


DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "models"
)


class FileArgumentParser(argparse.ArgumentParser):
    """Overwrites default ArgumentParser to better handle flag files."""

    def convert_arg_line_to_args(self, arg_line: str) -> Sequence[str]:
        """ Read from files where each line contains a flag and its value, e.g.
        '--flag value'. Also safely ignores comments denoted with '#' and
        empty lines.
        """

        # Remove any comments from the line.
        arg_line = arg_line.split('#')[0]

        # Escape if the line is empty.
        if not arg_line:
            return None

        # Separate flag and values.
        split_line = arg_line.strip().split(' ')

        # If there is actually a value, return the flag-value pair,
        if len(split_line) > 1:
            return [split_line[0], ' '.join(split_line[1:])]
        # Return just flag if there is no value.
        else:
            return split_line
        

def binary_to_bool(i: int) -> bool:
    if i != 0 and i != 1:
        raise ValueError("A binary integer (0 or 1) is expected.")
    return True if i else False


def set_if_absent(d: Dict[str, Any], key: str, default_value: Any) -> None:
    if key not in d:
        d[key] = default_value


def get_af3_args() -> Dict[str, Any]:
    """Creates a parser for AF3 and returns a dictionary of the parsed args.

    Returns:
        Dict[str, Any]: Dictionary mapping argument key to argument value.
    """
    parser = FileArgumentParser(
        description="Runner script for AlphaFold3.",
        fromfile_prefix_chars="@"
    )
    
    # Input and output paths.
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to the directory containing input JSON files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to a directory where the results will be saved."
    )
    
    # Model arguments.
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Path to the model to use for inference. Defaults to"
        f" {DEFAULT_MODEL_DIR}."
    )
    parser.add_argument(
        "--flash_attention_implementation",
        type=str,
        default="triton",
        choices=["triton", "cudnn", "xla"],
        help="Flash attention implementation to use. 'triton' and 'cudnn' uses"
        "a Triton and cuDNN flash attention implementation, respectively. The"
        " Triton kernel is fastest and has been tested more thoroughly. The"
        " Triton and cuDNN kernels require Ampere GPUs or later. 'xla' uses an"
        " XLA attention implementation (no flash attention) and is portable"
        " across GPU devices. Defaults to 'triton'."
    )
    
    # Control which stages to run.
    parser.add_argument(
        "--run_inference",
        type=int,
        default=1,
        help="Whether to run inference on the fold inputs. Defaults to 1"
        "(True)."
    )
    
    # Compilation arguments.
    parser.add_argument(
        "--jax_compilation_cache_dir",
        type=str,
        default=None,
        help="Path to a directory for the JAX compilation cache."
    )
    parser.add_argument(
        "--buckets",
        type=str,
        default="256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120",
        help="Strictly increasing order of token sizes for which to cache"
        " compilations (as comma-separated string). For any input with more"
        " tokens than the largest bucket size, a new bucket is created for"
        " exactly that number of tokens. Defaults to"
        " '256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120'."
    )
    
    args = parser.parse_args()
    
    # Reformat some of the arguments
    args.run_inference = binary_to_bool(args.run_inference)
    args.buckets = sorted([int(b) for b in args.buckets.split(',')])
    args.run_data_pipeline = False # Kuhlman Lab installation handles MSAs and templates differently
    
    return vars(args)


def set_json_defaults(json_str: str) -> str:
    """Loads a JSON-formatted string and applies some default values if they're not present.

    Args:
        json_str (str): A JSON-formatted string of fold inputs.

    Returns:
        str: A modified JSON-formatted string containing some extra defaults.
    """
    # Load the json_str
    raw_json = json.loads(json_str)
    
    if isinstance(raw_json, list):
        # AlphaFold Server JSON.
        # Don't apply the defaults to this format.
        pass
    else:
        # These defaults may need changed with future AF3 updates.
        set_if_absent(raw_json, 'dialect', 'alphafold3')
        set_if_absent(raw_json, 'version', 1)
        
        # Set default values for empty MSAs and templates
        for sequence in raw_json['sequences']:
            if "protein" in sequence:
                set_if_absent(sequence['protein'], 'unpairedMsa', '')
                set_if_absent(sequence['protein'], 'pairedMsa', '')
                set_if_absent(sequence['protein'], 'templates', [])
            elif 'rna' in sequence:
                set_if_absent(sequence['rna'], 'unpairedMsa', '')
                
        # Convert the dictionary back to a str
        json_str = json.dumps(raw_json)
    
    return json_str


def load_fold_inputs_from_path(json_path: pathlib.Path) -> Sequence[Input]:
    """Loads multiple fold inputs from a JSON path."""
    # Update the json defaults before parsing it.
    with open(json_path, 'r') as f:
        json_str = f.read()
    json_str = set_json_defaults(json_str)
    
    fold_inputs = []
    logging.info(
        'Detected %s is an AlphaFold 3 JSON since the top-level is not a list.',
        json_path,
    )
    # AlphaFold 3 JSON.
    try:
        fold_inputs.append(Input.from_json(json_str))
    except ValueError as e:
        raise ValueError(
            f'Failed to load fold input from {json_path}. The JSON at'
            f' {json_path} was detected to be an AlphaFold 3 JSON since the'
            ' top-level is not a list.'
        ) from e

    check_unique_sanitised_names(fold_inputs)

    return fold_inputs


def load_fold_inputs_from_dir(input_dir: pathlib.Path) -> Sequence[Input]:
    """Loads multiple fold inputs from all JSON files in a given input_dir.

    Args:
        input_dir: The directory containing the JSON files.

    Returns:
        The fold inputs from all JSON files in the input directory.

    Raises:
        ValueError: If the fold inputs have non-unique sanitised names.
    """
    fold_inputs = []
    for file_path in input_dir.glob('*.json'):
        if not file_path.is_file():
            continue

        fold_inputs.extend(load_fold_inputs_from_path(file_path))

    check_unique_sanitised_names(fold_inputs)

    return fold_inputs
