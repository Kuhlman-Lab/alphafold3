import json
import pathlib
import typing
from typing import Sequence, Union, Callable, Tuple, Dict, Any
from functools import partial
import textwrap
import jax

from alphafold3.jax.attention import attention
from alphafold3.model.diffusion import model as diffusion_model
from alphafold3.model.post_processing import post_process_inference_result

from af3_utils import load_fold_inputs_from_path, get_af3_args
from run_af3 import ModelRunner, make_model_config, predict_structure


def init_af3(proc_id: int, arg_file: str, lengths: Sequence[Union[int, Sequence[int]]]) -> Callable:
    args_dict = get_af3_args(arg_file)

    if args_dict['jax_compilation_cache_dir'] is not None:
        jax.config.update(
            'jax_compilation_cache_dir', args_dict['jax_compilation_cache_dir']
        )

    # Fail early on incompatible devices, only in init.
    gpu_devices = jax.local_devices(backend='gpu')
    if gpu_devices and float(gpu_devices[0].compute_capability) < 8.0:
        raise ValueError(
            'There are currently known unresolved numerical issues with using'
            ' devices with compute capability less than 8.0. See '
            ' https://github.com/google-deepmind/alphafold3/issues/59 for'
            ' tracking.'
        )

    # Keep notice in init function, only print for proc_id 0.
    if proc_id == 0:
        notice = textwrap.wrap(
            'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
            ' parameters are only available under terms of use provided at'
            ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
            ' If you do not agree to these terms and are using AlphaFold 3 derived'
            ' model parameters, cancel execution of AlphaFold 3 inference with'
            ' CTRL-C, and do not use the model parameters.',
            break_long_words=False,
            break_on_hyphens=False,
            width=80,
        )
        print('\n'.join(notice))

    devices = jax.local_devices(backend='gpu')
    model_runner = ModelRunner(
        model_class=diffusion_model.Diffuser,
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                attention.Implementation, args_dict["flash_attention_implementation"]
            ),
            num_diffusion_samples=1,
        ),
        device=devices[0],
        model_dir=pathlib.Path(args_dict["model_dir"]),
    )

    # Determine max length of sequences to use as bucket for model compilation
    max_length = 0
    for length in lengths:
        if not isinstance(length, int):
            length = max(length)
        if length > max_length:
            max_length = length
    buckets = (max_length,)

    # Use max_length to create a fake fold_input
    json_dict = {
        "name": "compilation_noodle",
        "sequences": [{
            "protein": {
                "id": ["A"],
                "sequence": "G" * max_length
            }
        }],
        "modelSeeds": ['42']
    }
    json_str = json.dumps(json_dict)
    fold_input = load_fold_inputs_from_path(json_str)[0]

    # Make folding prediction
    _ = model_runner.model_params
    _ = predict_structure(
        fold_input=fold_input,
        model_runner=model_runner, 
        buckets=buckets
    )

    return partial(run_af3, proc_id=proc_id, arg_file=arg_file, buckets=buckets, compiled_runner=model_runner)


def run_af3(json_str: str, proc_id: int, arg_file: str, buckets: Tuple[int], compiled_runner: ModelRunner) -> Sequence[Dict[str, Any]]:
    args_dict = get_af3_args(arg_file)
    
    if args_dict['jax_compilation_cache_dir'] is not None:
        jax.config.update(
            'jax_compilation_cache_dir', args_dict['jax_compilation_cache_dir']
        )

    # Convert json_str to fold_input and make prediction
    fold_input = load_fold_inputs_from_path(json_str)[0]
    all_inference_results = predict_structure(
        fold_input=fold_input,
        model_runner=compiled_runner, 
        buckets=buckets
    )

    # Convert results into list of dict for output
    processed_results = [
        post_process_inference_result(result.inference_results[0])
        for result in all_inference_results
    ]
    results_list = [
        {
            "seed": all_inference_results[i].seed,
            "cif_str": result.cif.decode("utf-8"),
            **json.loads(result.structure_confidence_summary_json.decode("utf-8")),
            **json.loads(result.structure_full_data_json.decode("utf-8")),

        }
        for i, result in enumerate(processed_results)
    ]

    return results_list
