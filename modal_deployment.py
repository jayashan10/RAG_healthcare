# # Serverless TensorRT-LLM (LLaMA 3 8B)

# In this example, we demonstrate how to use the TensorRT-LLM framework to serve Meta's LLaMA 3 8B model
# at very high throughput.

# We achieve a total throughput of over 25,000 output tokens per second on a single NVIDIA H100 GPU.
# At [Modal's on-demand rate](https://modal.com/pricing) of ~$4/hr, that's under $0.05 per million tokens --
# on auto-scaling infrastructure and served via a customizable API.

# ## Overview

# This guide is intended to document two things:
# the general process for building TensorRT-LLM on Modal
# and a specific configuration for serving the LLaMA 3 8B model.

# ### Build process

# Any given TensorRT-LLM service requires a multi-stage build process,
# starting from model weights and ending with a compiled engine.
# Because that process touches many sharp-edged high-performance components
# across the stack, it can easily go wrong in subtle and hard-to-debug ways
# that are idiosyncratic to specific systems.
# And debugging GPU workloads is expensive!

# This example builds an entire service from scratch, from downloading weight tensors
# to responding to requests, and so serves as living, interactive documentation of a TensorRT-LLM
# build process that works on Modal.

# ### Engine configuration

# TensorRT-LLM is the Lamborghini of inference engines: it achieves seriously
# impressive performance, but only if you tune it carefully.
# We carefully document the choices we made here and point to additional resources
# so you know where and how you might adjust the parameters for your use case.

# ## Installing TensorRT-LLM

# To run TensorRT-LLM, we must first install it. Easier said than done!

# In Modal, we define [container images](https://modal.com/docs/guide/custom-container) that run our serverless workloads.
# All Modal containers have access to GPU drivers via the underlying host environment,
# but we still need to install the software stack on top of the drivers, from the CUDA runtime up.

# We start from an official `nvidia/cuda` image,
# which includes the CUDA runtime & development libraries
# and the environment configuration necessary to run them.

from typing import Optional

import modal
import pydantic

tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    add_python="3.10",
).entrypoint([])

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).uv_pip_install(
    "tensorrt_llm==0.14.0",
    "pynvml<12",
    "cuda-python==12.9.1",
    "pyarrow<16.0.0",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
    extra_options="--index-strategy unsafe-best-match",
)

# ## Downloading the Model

MODEL_DIR = "/root/model/model_input"
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "b1532e4dee724d9ba63fe17496f298254d87ca64"


def download_model():
    import os

    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],
        revision=MODEL_REVISION,
    )
    move_cache()


MINUTES = 60
tensorrt_image = (
    tensorrt_image.uv_pip_install(
        "huggingface-hub==0.36.0",
        "requests~=2.32.2",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(
        download_model,
        timeout=20 * MINUTES,
    )
)

# ## Checkpoint Conversion

# Convert the model weights to TensorRT-LLM checkpoint format using the official convert_checkpoint.py script.
# This avoids the dependencies and complexity of the quantization step.

GPU_CONFIG = "H100:1"
CKPT_DIR = "/root/model/model_ckpt"
DTYPE = "float16"
N_GPUS = 1

CONVERT_CHECKPOINT_URL = "https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.14.0/examples/llama/convert_checkpoint.py"

tensorrt_image = tensorrt_image.run_commands(
    [
        f"wget {CONVERT_CHECKPOINT_URL} -O /root/convert_checkpoint.py",
        f"python /root/convert_checkpoint.py --model_dir={MODEL_DIR} --output_dir={CKPT_DIR} --tp_size={N_GPUS} --dtype={DTYPE}",
    ],
    gpu=GPU_CONFIG,
)

# ## Compiling the engine

MAX_INPUT_LEN, MAX_OUTPUT_LEN = 256, 256
MAX_NUM_TOKENS = 2**17
MAX_BATCH_SIZE = 1024
ENGINE_DIR = "/root/model/model_output"

SIZE_ARGS = f"--max_input_len={MAX_INPUT_LEN} --max_num_tokens={MAX_NUM_TOKENS} --max_batch_size={MAX_BATCH_SIZE}"

tensorrt_image = tensorrt_image.run_commands(
    [
        f"trtllm-build --checkpoint_dir {CKPT_DIR} --output_dir {ENGINE_DIR}"
        + f" --workers={N_GPUS}"
        + f" {SIZE_ARGS}",
    ],
    gpu=GPU_CONFIG,
).env({"TLLM_LOG_LEVEL": "INFO"})

# ## Serving inference

app = modal.App("TensorRT-LLaMa", image=tensorrt_image)


@app.cls(
    gpu=GPU_CONFIG,
    scaledown_window=10 * MINUTES,
    image=tensorrt_image,
)
class Model:
    @modal.enter()
    def load(self):
        import time

        print(f"{COLOR['HEADER']}Cold boot: spinning up TRT-LLM engine{COLOR['ENDC']}")
        self.init_start = time.monotonic_ns()

        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
        self.tokenizer.padding_side = "left"
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id

        runner_kwargs = dict(
            engine_dir=f"{ENGINE_DIR}",
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),
            max_output_len=MAX_OUTPUT_LEN,
        )

        self.model = ModelRunner.from_dir(**runner_kwargs)

        self.init_duration_s = (time.monotonic_ns() - self.init_start) / 1e9
        print(
            f"{COLOR['HEADER']}Cold boot finished in {self.init_duration_s}s{COLOR['ENDC']}"
        )

    @modal.method()
    def generate(self, prompts: list[str], settings=None):
        import time

        if settings is None or not settings:
            settings = dict(
                temperature=0.1,
                top_k=1,
                stop_words_list=None,
                repetition_penalty=1.1,
            )

        settings["max_new_tokens"] = MAX_OUTPUT_LEN
        settings["end_id"] = self.end_id
        settings["pad_id"] = self.pad_id

        num_prompts = len(prompts)

        if num_prompts > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {num_prompts} exceeds maximum of {MAX_BATCH_SIZE}"
            )

        print(
            f"{COLOR['HEADER']}Generating completions for batch of size {num_prompts}...{COLOR['ENDC']}"
        )
        start = time.monotonic_ns()

        parsed_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]

        print(
            f"{COLOR['HEADER']}Parsed prompts:{COLOR['ENDC']}",
            *parsed_prompts,
            sep="\n\t",
        )

        inputs_t = self.tokenizer(
            parsed_prompts, return_tensors="pt", padding=True, truncation=False
        )["input_ids"]

        print(f"{COLOR['HEADER']}Input tensors:{COLOR['ENDC']}", inputs_t[:, :8])

        outputs_t = self.model.generate(inputs_t, **settings)

        outputs_text = self.tokenizer.batch_decode(outputs_t[:, 0])

        responses = [
            extract_assistant_response(output_text) for output_text in outputs_text
        ]
        duration_s = (time.monotonic_ns() - start) / 1e9

        num_tokens = sum(map(lambda r: len(self.tokenizer.encode(r)), responses))

        for prompt, response in zip(prompts, responses):
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{prompt}",
                f"\n{COLOR['BLUE']}{response}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.05)

        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_ID} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second for batch of size {num_prompts} on {GPU_CONFIG}.{COLOR['ENDC']}"
        )

        return responses


@app.local_entrypoint()
def main():
    questions = [
        "What are you?",
        "What can you do?",
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
        "What are the differences between Javascript and Python?",
        "Explain the concept of inflation and its typical causes.",
        "What is the principle behind the operation of a microwave oven?",
        "How does Newton's third law apply to rocket propulsion?",
    ]

    model = Model()
    model.generate.remote(questions)


web_image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "fastapi[standard]==0.115.4",
    "pydantic==2.9.2",
    "starlette==0.41.2",
)


class GenerateRequest(pydantic.BaseModel):
    prompts: list[str]
    settings: Optional[dict] = None


@app.function(image=web_image)
@modal.fastapi_endpoint(
    method="POST", label=f"{MODEL_ID.lower().split('/')[-1]}-web", docs=True
)
def generate_web(data: GenerateRequest) -> list[str]:
    model = Model()
    return model.generate.remote(data.prompts, settings=data.settings)


COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}


def extract_assistant_response(output_text):
    parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>")

    if len(parts) > 1:
        response = parts[1].split("<|eot_id|>")[0].strip()
        response = response.replace("<|eot_id|>", "").strip()
        return response
    else:
        return output_text
