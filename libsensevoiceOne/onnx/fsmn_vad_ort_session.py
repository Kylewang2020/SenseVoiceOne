# -*- coding:utf-8 -*-
# @FileName  :fsmn_vad_ort_session.py.py
# @Time      :2024/8/31 16:45
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import logging
from pathlib import Path
from typing import List, Union

import numpy as np
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_available_providers,
    get_device,
)


class VadOrtInferRuntimeSession:
    def __init__(self, config, root_dir: Path):
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = "CUDAExecutionProvider"
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if (
            config["use_cuda"]
            and get_device() == "GPU"
            and cuda_ep in get_available_providers()
        ):
            EP_list = [(cuda_ep, config[cuda_ep])]
        EP_list.append((cpu_ep, cpu_provider_options))

        config["model_path"] = root_dir / str(config["model_path"])
        self._verify_model(config["model_path"])
        logging.debug(f"Loading onnx model at {str(config['model_path'])}")
        self.session = InferenceSession(
            str(config["model_path"]), sess_options=sess_opt, providers=EP_list
        )

        if config["use_cuda"] and cuda_ep not in self.session.get_providers():
            logging.warning(
                f"{cuda_ep} is not available for current env, "
                f"the inference part is automatically shifted to be "
                f"executed under {cpu_ep}.\n "
                "Please ensure the installed onnxruntime-gpu version"
                " matches your cuda and cudnn version, "
                "you can check their relations from the offical web site: "
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                RuntimeWarning,
            )

    def __call__(
        self, input_content: List[Union[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        if isinstance(input_content, list):
            input_dict = {
                "speech": input_content[0],
                "in_cache0": input_content[1],
                "in_cache1": input_content[2],
                "in_cache2": input_content[3],
                "in_cache3": input_content[4],
            }
        else:
            input_dict = {"speech": input_content}

        return self.session.run(None, input_dict)

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")
