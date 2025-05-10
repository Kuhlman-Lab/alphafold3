import os
import time
import argparse
import datetime
import tarfile
from typing import Sequence, Dict, Any, Union, Tuple, Optional, Iterator
import json
import pathlib
import logging
import requests
import itertools
from alphafold3.common.folding_input import Input, Template
from alphafold3.data import templates, structure_stores, msa_config
from alphafold3.structure import from_mmcif


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


def get_af3_parser() -> FileArgumentParser:
    """Creates a parser for AF3.

    Returns:
        FileArgumentParser: Argument parser for AF3.
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
    parser.add_argument(
        "--force_output_dir",
        type=int,
        default=0,
        help="Whether to force the output directory to be used even if it already"
        "exists and is non-empty."
    )
    
    # Output control.
    parser.add_argument(
        "--save_embeddings",
        type=int,
        default=0,
        help="Whether to save the final trunk single and pair embeddings in "
        "the output. Defaults to 0 (False)"
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
    parser.add_argument(
        "--num_recycles",
        type=int,
        default=10,
        help="Number of recycles to use during inference."
    )
    parser.add_argument(
        "--num_diffusion_samples",
        type=int,
        default=5,
        help="Number of diffusion samples to generate per seed. Defaults to 5."
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=None,
        help="Number of seeds to use for inference. If set, only a single"
        " seed must be provided in the input JSON. AlphaFold 3 will then"
        " generate random seeds in sequence, starting from the single seed"
        " specified in the input JSON. The full input JSON produced by"
        " AlphaFold 3 will include the generated random seeds. If not set,"
        " AlphaFold 3 will use the seeds as provided in the input JSON."
    )
    
    # Control which stages to run.
    parser.add_argument(
        "--run_inference",
        type=int,
        default=1,
        help="Whether to run inference on the fold inputs. Defaults to 1"
        "(True)."
    )
    
    # MMseqs for protein chains.
    parser.add_argument(
        "--run_mmseqs",
        action="store_true",
        help="If provided, MMseqs2 will be used to generate MSAs and "
        "templates for protein queries that have no custom inputs specified."
    )
    parser.add_argument(
        "--msa_mode",
        type=str,
        default="paired+unpaired",
        choices=["paired+unpaired", "unpaired", "paired"],
        help="The MSA mode to use. Options include ['paired+unpaired', 'unpaired', 'paired']. "
        "Defaults to 'paired+unpaired'."
    )
    parser.add_argument(
        "--pairing_strategy",
        type=str,
        default="greedy",
        choices=["greedy", "complete"],
        help="The strategy to use for pairing. Choices are ['greedy', 'complete']. "
        "Defaults to 'greedy'."
    )
    
    # Template search configuration.
    parser.add_argument(
        "--max_template_date",
        type=str,
        default='3000-01-01', # Set in far future.
        help="Maximum template release date to consider. Format: YYYY-MM-DD. "
        "All templates released after this date will be ignored. Controls also "
        "whether to allow use of model coordinates for a chemical component "
        "from the CCD if RDKit conformer generation fails and the component "
        "does not have ideal coordinates set. Only for components that have "
        "been released before this date the model coordinates can be used as "
        "a fallback."
    )

    # Conformer generation.
    parser.add_argument(
        "--conformer_max_iterations",
        type=int,
        default=None,  # Default to RDKit default parameters value.
        help="Optional override for maximum number of iterations to run for RDKit "
        "conformer search."
    )

    # Compilation and GPU arguments.
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
    parser.add_argument(
        "--cuda_compute_7x",
        type=int,
        default=0,
        help="If using a GPU with CUDA compute capability of 7.x, you must"
        " set this flag to 1. This will set "
        " XLA_FLAGS='--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'."
        " Defaults to 0 (False)."
    )
    parser.add_argument(
        "--gpu_device",
        type=int,
        default=0,
        help="Optional override for the GPU device to use for inference."
        " Defaults to the 1st GPU on the system. Useful on multi-GPU systems"
        " to pin each run to a specific GPU."
    )

    return parser


def get_af3_args(arg_file: Optional[str] = None) -> Dict[str, Any]:
    """Reformats args and returns a dictionary parsed args.

    Args:
        arg_file (str, optional): Path to the file containing argument key-
            value pairs. If None, then arguments are assumed to come from the
            command line. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary mapping argument key to argument value.
    """
    
    # Get the parser and args
    parser = get_af3_parser()
    if arg_file is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args([f'@{arg_file}'])
    
    # Reformat some of the arguments
    args.run_inference = binary_to_bool(args.run_inference)
    args.force_output_dir = binary_to_bool(args.force_output_dir)
    args.cuda_compute_7x = binary_to_bool(args.cuda_compute_7x)
    args.save_embeddings = binary_to_bool(args.save_embeddings)
    args.buckets = sorted([int(b) for b in args.buckets.split(',')])
    args.run_data_pipeline = False # Kuhlman Lab installation handles MSAs and templates differently
    
    # Check for ValueErrors in certain arguments
    if args.num_recycles < 1:
        raise ValueError("--num_recycles must be greater than or equal to 1.")
    if args.num_seeds is not None:
        if args.num_seeds < 1:
            raise ValueError("--num_seeds must be greater than or equal to 1.")
    
    return vars(args)


def _resolve_single_seq_msa(paired_msa_dict: Dict[str, str]) -> Dict[str, str]:
    """Checks if the paired MSA is a single sequence and if so returns an empty dict.

    Args:
        paired_msa_dict (Dict[str, str]): Dictionary containing the paired MSA.

    Returns:
        Dict[str, str]: Dictionary containing the resolved paired MSA.
    """
    single_seq_msas = []
    for seq in paired_msa_dict:
        if len(paired_msa_dict[seq].split('\n')) == 2:
            single_seq_msas.append(seq)
    
    if set(single_seq_msas) == set(paired_msa_dict.keys()):
        # If all MSAs are single sequences, return an empty dict
        return {}
    return paired_msa_dict


def _combine_msas(paired_msa_dict: Dict[str, str], unpaired_msa_dict: Dict[str, str]) -> Dict[str, str]:
    """Combines paired and unpaired MSAs into a single dictionary.

    Args:
        paired_msa_dict (Dict[str, str]): Dictionary containing the paired MSA.
        unpaired_msa_dict (Dict[str, str]): Dictionary containing the unpaired MSA.

    Returns:
        Dict[str, str]: Combined dictionary of MSAs.
    """
    assert set(paired_msa_dict.keys()) == set(unpaired_msa_dict.keys())

    def remove_first_seq(msa: str) -> str:
        """Removes the first sequence/description lines from the MSA."""
        lines = msa.split('\n')
        return '\n'.join(lines[2:])

    combined_msa_dict = {}
    for seq in paired_msa_dict:
        if seq in unpaired_msa_dict:
            combined_msa_dict[seq] = paired_msa_dict[seq] + '\n' + remove_first_seq(unpaired_msa_dict[seq])
        else:
            combined_msa_dict[seq] = paired_msa_dict[seq]
    
    return combined_msa_dict


def set_json_defaults(json_str: str, run_mmseqs: bool = False, output_dir: str = '', max_template_date: str = '3000-01-01', msa_mode: str = 'paired+unpaired', pairing_strategy: str = 'greedy') -> str:
    """Loads a JSON-formatted string and applies some default values if they're not present.

    Args:
        json_str (str): A JSON-formatted string of fold inputs.
        run_mmseqs (bool, optional): Whether to run MMseqs for MSAs and templates for 
            protein chains. Defaults to False.
        output_dir (str, optional): Place that'll store MMseqs2 MSAs and templates. Defaults to ''.
        max_template_date (str, optional): Maximum date for a template to be used. Defaults to '3000-01-01'.
        msa_mode (str, optional): The MSA mode to use. Options include ['paired+unpaired', 'unpaired', 'paired']. Defaults to 'paired+unpaired'. 
        pairing_strategy (str, optional): The strategy to use for pairing. Choices are ['greedy', 'complete']. Defaults to 'greedy'.

    Returns:
        str: A modified JSON-formatted string containing some extra defaults.
    """
    msa_mode = msa_mode.split('+')

    # Load the json_str
    raw_json = json.loads(json_str)
    
    if isinstance(raw_json, list):
        # AlphaFold Server JSON.
        # Don't apply the defaults to this format.
        return raw_json
    else:
        # These defaults may need changed with future AF3 updates.
        set_if_absent(raw_json, 'dialect', 'alphafold3')
        set_if_absent(raw_json, 'version', 3)

        # Resolve the ids in case if copies are provided.
        raw_json = _resolve_id_and_copies(raw_json)

        # Grab all of the protein sequences to run MMseqs on once.
        protein_seqs = [s['protein']['sequence'] for s in raw_json['sequences'] if 'protein' in s]
        if run_mmseqs and len(protein_seqs) > 0:
            # Run MMseqs2 on the protein sequences.
            a3m_paths_unpaired, template_dirs = run_mmseqs2(
                os.path.join(output_dir, f'mmseqs_{raw_json["name"]}_unpaired'),
                protein_seqs,
                use_templates=True
            )
            if 'paired' in msa_mode:
                # Run MMseqs2 on the protein sequences with pairing.
                a3m_paths_paired, _ = run_mmseqs2(
                    os.path.join(output_dir, f'mmseqs_{raw_json["name"]}_paired'),
                    protein_seqs,
                    use_pairing=True,
                    pairing_strategy=pairing_strategy,
                    use_templates=False
                )
            else:
                a3m_paths_paired = [""]

            # Set the MSAs and templates.
            for i, sequence in enumerate(raw_json['sequences']):
                if 'protein' in sequence:
                    if 'unpaired' in msa_mode:
                        if 'unpairedMsaPath' not in sequence['protein']:
                            set_if_absent(sequence['protein'], 'unpairedMsa', a3m_paths_unpaired[i])
                    if a3m_paths_paired != [""] and 'pairedMsaPath' not in sequence['protein']:
                        set_if_absent(sequence['protein'], 'pairedMsa', a3m_paths_paired[i])
                    set_if_absent(sequence['protein'], 'templates', [] if template_dirs[i] is None else template_dirs[i])          

        # Set default values for empty MSAs and templates
        for sequence in raw_json['sequences']:
            if "protein" in sequence:
                if ('unpairedMsa' in sequence['protein'] or 'unpairedMsaPath' in sequence['protein']) and 'templates' in sequence['protein']:
                    # If both unpairedMsa (or unpairedMsaPath) and templates are provided, use them and maybe set pairedMsa
                    pass
                else:
                    # Set empty values.
                    set_if_absent(sequence['protein'], 'unpairedMsa', '')
                    set_if_absent(sequence['protein'], 'templates', [])

                if 'unpairedMsaPath' in sequence['protein']:
                    # Move unpairedMsaPath to unpairedMsa for unified parsing of MSAs
                    sequence['protein']['unpairedMsa'] = sequence['protein']['unpairedMsaPath']
                    del sequence['protein']['unpairedMsaPath']

                if 'paired' in msa_mode and 'pairedMsa' in sequence['protein']:
                    if sequence['protein']['pairedMsa'] != "" and os.path.exists(sequence['protein']['pairedMsa']):
                        # If pairedMsa isn't empty and is a path that exists, parse it as a custom MSA
                        msa_dict = get_custom_msa_dict(sequence['protein']['pairedMsa'])
                        msa_dict = _resolve_single_seq_msa(msa_dict)
                        del sequence['protein']['pairedMsa']

                        if 'unpaired' not in msa_mode:
                            # If not using unpaired MSAs, set the pairedMsa to the unpaired MSA
                            sequence['protein']['unpairedMsa'] = msa_dict.get(sequence['protein']['sequence'], "")
                else:
                    msa_dict = {}

                if 'unpaired' in msa_mode:
                    if sequence['protein']['unpairedMsa'] != "" and os.path.exists(sequence['protein']['unpairedMsa']):
                        # If unpairedMsa isn't empty and is a path that exists, parse it as a custom MSA
                        if msa_dict == {}:
                            msa_dict = get_custom_msa_dict(sequence['protein']['unpairedMsa'])
                        else:
                            # If msa_dict isn't {}, then a paired MSA was found. We need to combine
                            # it with the unpaired MSA since msa_mode must be paired+unpaired
                            unpaired_msa_dict = get_custom_msa_dict(sequence['protein']['unpairedMsa'])
                            msa_dict = _combine_msas(msa_dict, unpaired_msa_dict)
                        sequence['protein']['unpairedMsa'] = msa_dict.get(sequence['protein']['sequence'], "")
                
                if sequence['protein']['templates'] != [] and type(sequence['protein']['templates']) == str:
                    if os.path.exists(sequence['protein']['templates']):
                        # If templates isn't empty and is a path that exists, parse it as custom templates
                        template_hits = get_custom_template_hits(
                            sequence['protein']['sequence'], 
                            sequence['protein']['unpairedMsa'], 
                            sequence['protein']['templates'],
                            max_template_date=max_template_date
                        )
                        sequence['protein']['templates'] = template_hits

                # Make sure pairedMsa is set no matter what
                set_if_absent(sequence['protein'], 'pairedMsa', '')
            elif 'rna' in sequence:
                set_if_absent(sequence['rna'], 'unpairedMsa', '')
    
    return raw_json


def _resolve_id_and_copies(raw_json: Dict[str, Any]) -> Dict[str, Any]:
    # Resolve the ids in case if copies are provided.
    ids_present = set()
    copies = 0
    for sequence in raw_json['sequences']:
        check = [('id' not in sequence[k] or 'copies' not in sequence[k]) for k in sequence]
        if check == [False]:
            raise ValueError("Both 'copies' and 'id' cannot be present in the JSON.")
        ids = [sequence[k].get('id', []) for k in sequence]
        ids_present = ids_present.union(set([e for i in ids for e in i]))
        copies += sum([sequence[k].get('copies', 0) for k in sequence])
    total_chains = len(ids_present) + copies
    if len(ids_present) != total_chains:
        # Generate chain id combinations
        alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        max_combi = 1 + total_chains // len(alphabet)
        combinations = []
        for length in range(1, max_combi + 1):
            current_combinations = [''.join(combo) for combo in itertools.product(alphabet, repeat=length)]
            combinations.extend(current_combinations)
        
        # Grab ids to assign based on those already present.
        possible_ids = [i for i in combinations if i not in ids_present]
        remaining_ids = possible_ids[:total_chains - len(ids_present)]

        # Assign the missing ids
        for sequence in raw_json['sequences']:
            for k in sequence:
                if "copies" in sequence[k]:
                    sequence[k]["id"] = [remaining_ids.pop(0) for _ in range(sequence[k]["copies"])]
                    sequence[k].pop("copies")

    return raw_json
        

def load_fold_inputs_from_path(json_path: Union[pathlib.Path, str], run_mmseqs: bool = False, output_dir: str = '', max_template_date: str = '3000-01-01', msa_mode: str = 'paired+unpaired', pairing_strategy: str = 'greedy') -> Iterator[Input]:
    """Loads multiple fold inputs from a JSON path (or string of a JSON).

    Args:
        json_path (Union[pathlib.Path, str]): Either the path to the JSON file or the string 
            corresponding to it.
        run_mmseqs (bool, optional): Whether to run MMseqs on protein chains. Defaults to False.
        output_dir (str, optional): Place that'll store MMseqs MSAs and templates. Defaults to ''.
        max_template_date (str, optional): Maximum date for a template to be used. Defaults to '3000-01-01'.
        msa_mode (str, optional): The MSA mode to use. Options include ['paired+unpaired', 'unpaired', 'paired']. Defaults to 'paired+unpaired'.
        pairing_strategy (str, optional): The strategy to use for pairing. Choices are ['greedy', 'complete']. Defaults to 'greedy'.

    Raises:
        ValueError: Fails if we cannot load json_path as an AlphaFold3 JSON

    Yields:
        The folding inputs.
    """
    # Update the json defaults before parsing it.
    if not isinstance(json_path, str):
        with open(json_path, 'r') as f:
            json_str = f.read()
    else:
        json_str = json_path
    raw_json = set_json_defaults(json_str, run_mmseqs, output_dir, max_template_date, msa_mode, pairing_strategy)
    json_str = json.dumps(raw_json)

    if isinstance(raw_json, list):
        # AlphaFold Server JSON.
        logging.info('Loading %d fold jobs from %s', len(raw_json), json_path)
        for fold_job_idx, fold_job in enumerate(raw_json):
            try:
                yield Input.from_alphafoldserver_fold_job(fold_job)
            except ValueError as e:
                raise ValueError(
                    f'Failed to load fold job {fold_job_idx} from {json_path}. The JSON'
                    ' was detected to be the AlphaFold Server dialect.'
                ) from e
    else:
        # AlphaFold 3 JSON.
        try:
            yield Input.from_json(json_str, json_path if not isinstance(json_path, str) else None)
        except ValueError as e:
            raise ValueError(
                f'Failed to load fold input from {json_path}. The JSON was detected'
                f' to be the AlphaFold 3 dialect.'
            ) from e


def load_fold_inputs_from_dir(input_dir: pathlib.Path, run_mmseqs: bool = False, output_dir: str = '', max_template_date: str = '3000-01-01', msa_mode: str = 'paired+unpaired', pairing_strategy: str = 'greedy') -> Iterator[Input]:
    """Loads multiple fold inputs from all JSON files in a given input_dir.

    Args:
        input_dir (pathlib.Path): The directory containing the JSON files.
        run_mmseqs (bool, optional): Whether to run MMseq2 on protein chains. Defaults to False.
        output_dir (str, optional): Place that'll store MMseqs2 MSAs and templates. Defaults to ''.
        max_template_date (str, optional): Maximum date for a template to be used. Defaults to '3000-01-01'.
        msa_mode (str, optional): The MSA mode to use. Options include ['paired+unpaired', 'unpaired', 'paired']. Defaults to 'paired+unpaired'.
        pairing_strategy (str, optional): The strategy to use for pairing. Choices are ['greedy', 'complete']. Defaults to 'greedy'.

    Yields:
        The fold inputs from all JSON files in the input directory.
    """
    for file_path in sorted(input_dir.glob('*.json')):
        if not file_path.is_file():
            continue

        yield from load_fold_inputs_from_path(file_path, run_mmseqs, output_dir, max_template_date, msa_mode, pairing_strategy)


def run_mmseqs2(
        prefix: str,
        sequences: Union[Sequence[str], str],
        use_env: bool = True,
        use_templates: bool = False,
        num_templates: int = 20,
        use_pairing: bool = False,
        pairing_strategy: str = 'greedy',
        host_url: str = 'https://api.colabfold.com'
        ) -> Tuple[Sequence[str], Sequence[Optional[str]]]:
    """Computes MSAs and templates by querying ColabFold MMseqs2 server.

    Args:
        prefix (str): Prefix for the output directory that'll store MSAs and templates.
        sequences (Union[Sequence[str], str]): The sequence(s) that'll be used as queries for MMseqs
        use_env (bool, optional): Whether to include the environmental database in the search. Defaults to True.
        use_templates (bool, optional): Whether to search for templates. Defaults to False.
        num_templates (int, optional): How many templates to search for. Defaults to 20.
        use_pairing (bool, optional): Whether to generate a species-paired MSA. Defaults to False.
        pairing_strategy (str, optional): The strategy to use for pairing. Choices are ['greedy', 'complete']. Defaults to 'greedy'.
        host_url (str, optional): URL to ColabFold MMseqs server. Defaults to 'https://api.colabfold.com'.

    Raises:
        Exception: Errors related to MMseqs. Sometimes these can be solved by simply trying again.

    Returns:
        Tuple[Sequence[str], Sequence[Optional[str]]]: A Tuple of (MSAs, templates). MSAs are the paths to the
            MMseqs MSA generated for each sequence; similarly templates point to a directory of templates.
    """
    submission_endpoint = 'ticket/msa' if not use_pairing else 'ticket/pair'
    og_sequences = sequences
    
    def submit(seqs: Sequence[str], mode: str, N=101) -> Dict[str, str]:
        """ Submits a query of sequences to MMseqs2 API. """

        n, query = N, ''
        for seq in seqs:
            query += f'>{n}\n{seq}\n'
            n += 1

        res = requests.post(f'{host_url}/{submission_endpoint}',
                            data={'q': query, 'mode': mode})
        try:
            out = res.json()
        except ValueError:
            out = {'status': 'ERROR'}

        return out

    def status(ID: int) -> Dict[str, str]:
        """ Obtains the status of a submitted query. """
        res = requests.get(f'{host_url}/ticket/{ID}')
        try:
            out = res.json()
        except ValueError:
            out = {'status': 'ERROR'}

        return out

    def download(ID: int, path: str) -> None:
        """ Downloads the completed MMseqs2 query. """
        res = requests.get(f'{host_url}/result/download/{ID}')
        with open(path, 'wb') as out:
            out.write(res.content)

    # Make input sequence a list if not already.
    sequences = [og_sequences] if isinstance(og_sequences, str) else og_sequences
            
    # Set the mode for MMseqs2.
    mode = 'env' if use_env else 'all'
    if use_pairing:
        use_templates = False
        mode = 'pair'
        if pairing_strategy == 'greedy':
            mode += 'greedy'
        elif pairing_strategy == 'complete':
            mode += 'complete'
        else:
            raise ValueError(f"Unrecognized pairing strategy: {pairing_strategy}. "
                             "Must be either 'greedy' or 'complete'.")
        if use_env:
            mode += '-env'

    # Deduplicate and keep track of order.
    unique_seqs = []
    [unique_seqs.append(seq) for seq in sequences if seq not in unique_seqs]
    if len(unique_seqs) == 1 and use_pairing:
        # If only one sequence is provided, pairing is not necessary.
        return [""], [None]
    N, REDO = 101, True
    Ms = [N + unique_seqs.index(seq) for seq in sequences]
    
    # Set up output path after potentially exiting.
    out_path = f'{prefix}_{mode}'
    os.makedirs(out_path, exist_ok=True)
    tar_gz_file = os.path.join(out_path, 'out.tar.gz')

    # Call MMseqs2 API.
    if not os.path.isfile(tar_gz_file):
        while REDO:
            # Resubmit job until it goes through
            out = submit(seqs=unique_seqs, mode=mode, N=N)
            while out['status'] in ['UNKNOWN', 'RATELIMIT']:
                # Resubmit
                time.sleep(5)
                out = submit(seqs=unique_seqs, mode=mode, N=N)

            if out['status'] == 'ERROR':
                raise Exception('MMseqs2 API is giving errors. Please confirm '
                                'your input is a valid protein sequence. If '
                                'error persists, please try again in an hour.')

            if out['status'] == 'MAINTENANCE':
                raise Exception('MMseqs2 API is undergoing maintenance. Please '
                                'try again in a few minutes.')
                
            # Wait for job to finish
            ID = out['id']
            while out['status'] in ['UNKNOWN', 'RUNNING', 'PENDING']:
                time.sleep(5)
                out = status(ID)

            if out['status'] == 'COMPLETE':
                REDO = False

            if out['status'] == 'ERROR':
                REDO = False
                raise Exception('MMseqs2 API is giving errors. Please confirm '
                                'your input is a valid protein sequence. If '
                                'error persists, please try again in an hour.')
        # Download results
        download(ID, tar_gz_file)

    # Get and extract a list of .a3m files.
    if use_pairing:
        a3m_files = [os.path.join(out_path, 'pair.a3m')]
    else:
        a3m_files = [os.path.join(out_path, 'uniref.a3m')]
        if use_env:
            a3m_files.append(
                os.path.join(out_path, 'bfd.mgnify30.metaeuk30.smag30.a3m'))
    if not os.path.isfile(a3m_files[0]):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(out_path)

    # Get templates if necessary. 
    if use_templates:
        templates = {}
        
        # Read MMseqs2 template outputs and sort templates based on query seq.
        with open(os.path.join(out_path, 'pdb70.m8'), 'r') as f:
            for line in f:
                p = line.rstrip().split()
                M, pdb = p[0], p[1]
                M = int(M)
                if M not in templates:
                    templates[M] = []
                templates[M].append(pdb)

        # Obtain template structures and data files
        template_paths = {}
        for k, TMPL in templates.items():
            TMPL_PATH = os.path.join(prefix+'_'+mode, f'templates_{k}')
            if not os.path.isdir(TMPL_PATH):
                os.mkdir(TMPL_PATH)
                TMPL_LINE = ','.join(TMPL[:num_templates])
                # Obtain the .cif and data files for the templates
                os.system(
                    f'curl -s -L {host_url}/template/{TMPL_LINE} '
                    f'| tar xzf - -C {TMPL_PATH}/')
                # Rename data files
                os.system(
                    f'cp {TMPL_PATH}/pdb70_a3m.ffindex '
                    f'{TMPL_PATH}/pdb70_cs219.ffindex')
                os.system(f'touch {TMPL_PATH}/pdb70_cs219.ffdata')
            template_paths[k] = TMPL_PATH

    # Gather .a3m lines.
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        with open(a3m_file, 'r') as f:
            for line in f:
                if len(line) > 0:
                    # Replace NULL values
                    if '\x00' in line:
                        line = line.replace('\x00', '')
                        update_M = True
                    if line.startswith('>') and update_M:
                        M = int(line[1:].rstrip())
                        update_M = False
                        if M not in a3m_lines:
                            a3m_lines[M] = []
                    a3m_lines[M].append(line)

    # Save the complete MSAs
    a3m_lines = [''.join(a3m_lines[n]) for n in Ms]
    a3m_paths = []
    for i, n in enumerate(Ms):
        a3m_path = os.path.join(out_path, f"mmseqs_{n}.a3m")
        a3m_paths.append(a3m_path)
        with open(a3m_path, 'w') as f:
            f.write(a3m_lines[i])

    if use_templates:
        template_paths_ = []
        for n in Ms:
            if n not in template_paths:
                template_paths_.append(None)
            else:
                template_paths_.append(template_paths[n])
        template_paths = template_paths_
    else:
        template_paths = []
        for n in Ms:
            template_paths.append(None)

    if isinstance(og_sequences, str):
        return (a3m_paths[0], template_paths[0])
    else:
        return (a3m_paths, template_paths)


def get_custom_msa_dict(custom_msa_path: str) -> Dict[str, str]:
    """Reads a custom MSA and returns a dict mapping query to MSA.

    Args:
        custom_msa_path (str): Path to the custom MSA.

    Raises:
        ValueError: If the MSA path isn't an .a3m file.
        ValueError: If no MSAs were parsed from the file.

    Returns:
        Dict[str, str]: Mapping from query sequence to MSA.
    """
    assert os.path.exists(custom_msa_path)
    
    custom_msa_dict = {}
    extension = custom_msa_path.split('.')[-1]
    if extension == 'a3m':
        with open(custom_msa_path, 'r') as f:
            a3m_lines = f.read()
        
        # Parse the a3m lines and grab sequences, splitting by the first sequence
        update_seq, seq = True, None
        capture_seq = False
        for line in a3m_lines.splitlines():
            if len(line) > 0:
                if '\x00' in line:
                    line = line.replace('\x00', '')
                    update_seq = True
                if line.startswith('>') and update_seq:
                    capture_seq = True
                    update_seq = False
                    header = line
                    continue
                if capture_seq:
                    seq = line.rstrip()
                    capture_seq = False
                    if seq not in custom_msa_dict:
                        custom_msa_dict[seq] = [header]
                    else:
                        continue

                if len(line) > 0:
                    custom_msa_dict[seq].append(line)
    else:
        raise ValueError(f"Unrecognized extension for custom MSA: {custom_msa_path}. We currently only accept .a3m")
    
    # Combine MSA lines into single string
    for seq in custom_msa_dict:
        custom_msa_dict[seq] = '\n'.join(custom_msa_dict[seq])

    if custom_msa_dict == {}:
        raise ValueError(
            f'No custom MSAs detected in {custom_msa_path}. Double-check the '
            f'path or no not provide the --custom_msa_path argument. Note that'
            f'custom MSAs must be in .a3m format')
    
    return custom_msa_dict


def get_custom_template_hits(
        query_seq: str, 
        unpaired_msa: str, 
        template_path: str,
        max_template_date: str,
    ) -> Sequence[Dict[str, Union[str, Sequence[int]]]]:
    """Parses .cif files for templates to a query seq and its MSA. This also formats them for AF3 

    Args:
        query_seq (str): Query sequence for templates.
        unpaired_msa (str): Query's MSA for the HMM.
        template_path (str): Path to directory containing .cif files to search.
        max_template_date (str): Maximum date allowed for a template to be used.

    Returns:
        Sequence[Dict[str, Union[str, Sequence[int]]]]: A list of dictionaries of templates 
            formatted for AF3 JSON input.
    """

    # Make a fake template database
    db_file = os.path.join(template_path, 'template_db.a3m')
    cif_files = pathlib.Path(template_path).glob("*.cif")
    store_mapping = {}
    with open(db_file, 'w') as a3m:
        for cif_file in cif_files:
            pdb_name = os.path.basename(cif_file)[:-4]
            with open(cif_file) as f:
                cif_string = f.read()
            struc = from_mmcif(cif_string, name=pdb_name)
            if struc.release_date is None:
                # If no release date, assume it's safe to use.
                # I know its bad practice, but I'm setting the private variable
                struc._release_date = datetime.date.fromisoformat(max_template_date)
            chain_map = struc.polymer_author_chain_single_letter_sequence(rna=False, dna=False)
            for ch in chain_map:
                a3m_str = f">{pdb_name}_{ch} length:{len(chain_map[ch])}\n{chain_map[ch]}\n"
                a3m.write(a3m_str)
            cif_str = struc.to_mmcif()
            store_mapping[pdb_name] = cif_str

    # If the MSA is empty, make sure it at least has the query sequence.
    if unpaired_msa == "":
        unpaired_msa = f">query\n{query_seq}"
                
    # Reformat the unpaired_msa so that the descriptions have no spaces in them
    unpaired_msa_lines = unpaired_msa.splitlines()
    for i in range(0, len(unpaired_msa_lines), 2):
        unpaired_msa_lines[i] = unpaired_msa_lines[i].split('\t')[0]
    unpaired_msa = '\n'.join(unpaired_msa_lines)

    # Create the templates object.
    templates_obj = templates.Templates.from_seq_and_a3m(
        query_sequence=query_seq,
        msa_a3m=unpaired_msa,
        max_template_date=datetime.date.fromisoformat(max_template_date),
        database_path=db_file,
        hmmsearch_config=msa_config.HmmsearchConfig(
            hmmsearch_binary_path=os.path.join(os.environ["CONDA_PREFIX"], "bin", "hmmsearch"),
            hmmbuild_binary_path=os.path.join(os.environ["CONDA_PREFIX"], "bin", "hmmbuild"),
            filter_f1=0.1,
            filter_f2=0.1,
            filter_f3=0.1,
            e_value=100,
            inc_e=100,
            dom_e=100,
            incdom_e=100    
        ),
        max_a3m_query_sequences=None,
        structure_store=structure_stores.StructureStore(store_mapping)
    )
    
    # Filter templates.
    templates_obj = templates_obj.filter(
        max_subsequence_ratio=1.00, # Keep perfect matches
        min_align_ratio=0.1,
        min_hit_length=10,
        deduplicate_sequences=True,
        max_hits=4
    )
    
    # Get both hits and their structures
    template_list = [
        Template(
            mmcif=struc.to_mmcif(),
            query_to_template_map=hit.query_to_hit_mapping,
        )
        for hit, struc in templates_obj.get_hits_with_structures()
    ]

    # Decompose templates into expected json format
    template_json = [
        {
            "mmcif": t._mmcif,
            "queryIndices": list(t.query_to_template_map.keys()),
            "templateIndices": list(t.query_to_template_map.values())
        }
        for t in template_list
    ]
    
    return template_json
