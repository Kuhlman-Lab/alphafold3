import os
import json
import pathlib
import typing
from typing import Sequence, Union, Callable, Tuple, Dict, Any
from functools import partial
import textwrap
import jax

from alphafold3.jax.attention import attention
from alphafold3.model.post_processing import post_process_inference_result

from af3_utils import load_fold_inputs_from_path, get_af3_args
from run_af3 import ModelRunner, make_model_config, predict_structure


def init_af3(proc_id: int, arg_file: str, lengths: Sequence[Union[int, Sequence[int]]]) -> Callable:
    # Work around for a known XLA issue:
    # https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md#compilation-time-workaround-with-xla-flags
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
    
    args_dict = get_af3_args(arg_file)

    if args_dict['jax_compilation_cache_dir'] is not None:
        jax.config.update(
            'jax_compilation_cache_dir', args_dict['jax_compilation_cache_dir']
        )

    # Fail early on incompatible devices, only in init.
    gpu_devices = jax.local_devices(backend='gpu')
    if gpu_devices:
        compute_capability = float(gpu_devices[args_dict["gpu_device"]].compute_capability)
        if compute_capability < 6.0:
            raise ValueError(
                'AlphaFold 3 requires at least GPU compute capability 6.0 (see'
                ' https://developer.nvidia.com/cuda-gpus).'
            )
        elif 7.0 <= compute_capability < 8.0:
            xla_flags = os.environ.get('XLA_FLAGS')
            required_flag = '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
            if not xla_flags or required_flag not in xla_flags:
                raise ValueError(
                    'For devices with GPU compute capability 7.x (see'
                    ' https://developer.nvidia.com/cuda-gpus), you must include'
                    ' the --cuda_compute_7x flag.'
                )
            if args_dict["flash_attention_implementation"] != "xla":
                raise ValueError(
                    'For devices with GPU compute capability 7.x (see '
                    ' https://developer.nvidia.com/cuda-gpus) the '
                    ' --flash_attention_implementation must be set to "xla".'
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
        print('\n' + '\n'.join(notice) + '\n')

    devices = jax.local_devices(backend='gpu')
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                attention.Implementation, args_dict["flash_attention_implementation"]
            ),
            num_diffusion_samples=1, # To reduce time
            num_recycles=1, # To reduce time
        ),
        device=devices[args_dict["gpu_device"]],
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
    fold_input = [i for i in load_fold_inputs_from_path(json_str)][0]

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
    fold_input = [i for i in load_fold_inputs_from_path(json_str)][0]
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
