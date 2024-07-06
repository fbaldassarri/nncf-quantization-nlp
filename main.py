#
# License: GPL 3
#

import os
from os import PathLike
import time
from pathlib import Path
from zipfile import ZipFile
from typing import Iterable, Any

import datasets
import evaluate
import numpy 
import nncf
from nncf.parameters import ModelType
import openvino 
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Define routine to download the files
# Note: the file will be saved on the local filesystem to the current directory by default. Define a `directory` as needed to change this behaviour. If a `filename` is not given, the filename of the URL will be used.
#
#    :param url: URL that points to the file to download
#    :param filename: Name of the local file to save. Should point to the name of the file only,
#                     not the full path. If None the filename from the url will be used
#    :param directory: Directory to save the file to. Will be created if it doesn't exist
#                      If None the file will be saved to the current working directory
#    :param show_progress: If True, show an TQDM ProgressBar
#    :param silent: If True, do not print a message if the file already exists
#    :param timeout: Number of seconds before cancelling the connection attempt
#    :return: path to downloaded file
#
# Source: https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py
#

import urllib.parse

def download_file(
    url: PathLike,
    filename: PathLike = None,
    directory: PathLike = None,
    show_progress: bool = True,
    silent: bool = False,
    timeout: int = 10,
) -> PathLike:

    from tqdm import tqdm
    import requests

    filename = filename or Path(urllib.parse.urlparse(url).path).name
    chunk_size = 16384  # make chunks bigger so that not too many updates are triggered for Jupyter front-end

    filename = Path(filename)
    if len(filename.parts) > 1:
        raise ValueError(
            "`filename` should refer to the name of the file, excluding the directory. "
            "Use the `directory` parameter to specify a target directory for the downloaded file."
        )

    # create the directory if it does not exist, and add the directory to the filename
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / Path(filename)

    try:
        response = requests.get(url=url, headers={"User-agent": "Mozilla/5.0"}, stream=True)
        response.raise_for_status()
    except (
        requests.exceptions.HTTPError
    ) as error:  # For error associated with not-200 codes. It will output something like: "404 Client Error: Not Found for url: {url}"
        raise Exception(error) from None
    except requests.exceptions.Timeout:
        raise Exception(
            "Connection timed out. If you access the internet through a proxy server, please "
            "make sure the proxy is set in the shell from where you launched Jupyter."
        ) from None
    except requests.exceptions.RequestException as error:
        raise Exception(f"File downloading failed with error: {error}") from None

    # download the file, if it does not exist; or download it again if it exists but has an incorrect file size
    filesize = int(response.headers.get("Content-length", 0))
    if not filename.exists() or (os.stat(filename).st_size != filesize):
        with tqdm(
            total=filesize,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=str(filename),
            disable=not show_progress,
        ) as progress_bar:
            with open(filename, "wb") as file_object:
                for chunk in response.iter_content(chunk_size):
                    file_object.write(chunk)
                    progress_bar.update(len(chunk))
                    progress_bar.refresh()
    else:
        if not silent:
            print(f"'{filename}' already exists.")

    response.close()

    return filename.resolve()

# Set the data and model directories, source URL and the filename of the model.
MODEL_DIR = "model"
MODEL_LINK = "https://download.pytorch.org/tutorial/MRPC.zip"
FILE_NAME = MODEL_LINK.split("/")[-1]
PRETRAINED_MODEL_DIR = os.path.join(MODEL_DIR, "MRPC")

os.makedirs(MODEL_DIR, exist_ok=True)

# Download and unpack pre-trained BERT model for MRPC by PyTorch 

download_file(MODEL_LINK, directory=MODEL_DIR, show_progress=True)
with ZipFile(f"{MODEL_DIR}/{FILE_NAME}", "r") as zip_ref:
    zip_ref.extractall(MODEL_DIR)

# Remove MRPC.zip
if os.path.exists("model/MRPC.zip"):
    os.remove("model/MRPC.zip")
else:
    print("MRPC.zip: the file does not exist.")

# Convert the original PyTorch model to the OpenVINO Intermediate Representation (OpenVINO IR)
# From OpenVINO 2023.0, we can directly convert a model from the PyTorch format to the OpenVINO IR format using model conversion API. Following PyTorch model formats are supported:
# - `torch.nn.Module`
# - `torch.jit.ScriptModule`
# - `torch.jit.ScriptFunction`

MAX_SEQ_LENGTH = 128
input_shape = openvino.PartialShape([1, -1])
ir_model_xml = Path(MODEL_DIR) / "bert_mrpc.xml"
core = openvino.Core()

torch_model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_DIR)
torch_model.eval

input_info = [
    ("input_ids", input_shape, numpy.int64),
    ("attention_mask", input_shape, numpy.int64),
    ("token_type_ids", input_shape, numpy.int64),
]
default_input = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64)
inputs = {
    "input_ids": default_input,
    "attention_mask": default_input,
    "token_type_ids": default_input,
}

# Convert the PyTorch model to OpenVINO IR FP32.
if not ir_model_xml.exists():
    model = openvino.convert_model(torch_model, example_input=inputs, input=input_info)
    openvino.save_model(model, str(ir_model_xml))
else:
    model = core.read_model(ir_model_xml)

# Preparing the Dataset, Splitting
def create_data_source():
    raw_dataset = datasets.load_dataset("glue", "mrpc", split="validation")
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)

    def _preprocess_fn(examples):
        texts = (examples["sentence1"], examples["sentence2"])
        result = tokenizer(*texts, padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True)
        result["labels"] = examples["label"]
        return result

    processed_dataset = raw_dataset.map(_preprocess_fn, batched=True, batch_size=1)

    return processed_dataset

data_source = create_data_source()

# Optimize model using NNCF Post-training Quantization API
INPUT_NAMES = [key for key in inputs.keys()]

def transform_fn(data_item):
    """
    Extract the model's input from the data item.
    The data item here is the data item that is returned from the data source per iteration.
    This function should be passed when the data item cannot be used as model's input.
    """
    inputs = {name: numpy.asarray([data_item[name]], dtype=numpy.int64) for name in INPUT_NAMES}
    return inputs

calibration_dataset = nncf.Dataset(data_source, transform_fn)

# Quantize the model. By specifying model_type, we specify additional transformer patterns in the model.
quantized_model = nncf.quantize(model, calibration_dataset, model_type=ModelType.TRANSFORMER)
compressed_model_xml = Path(MODEL_DIR) / "quantized_bert_mrpc.xml"
openvino.save_model(quantized_model, compressed_model_xml)

# Load and Test OpenVINO Model 
# Compile the model for a specific device.
compiled_quantized_model = core.compile_model(model=quantized_model, device_name="CPU")
output_layer = compiled_quantized_model.outputs[0]

# The Data Source returns a pair of sentences (indicated by `sample_idx`) and the inference compares these sentences and outputs whether their meaning is the same. You can test other sentences by changing `sample_idx` to another value (from 0 to 407).
sample_idx = 5
sample = data_source[sample_idx]
inputs = {k: torch.unsqueeze(torch.tensor(sample[k]), 0) for k in ["input_ids", "token_type_ids", "attention_mask"]}

result = compiled_quantized_model(inputs)[output_layer]
result = numpy.argmax(result)

print(f"Sentence 1: {sample['sentence1']}")
print(f"Sentence 2: {sample['sentence2']}")
print(f"Have the same meaning? {'Yes' if result == 1 else 'No'}")

# Compare F1-score of FP32 and INT8 models
def validate(model: openvino.Model, dataset: Iterable[Any]) -> float:
    """
    Evaluate the model on GLUE dataset.
    Returns F1 score metric.
    """
    compiled_model = core.compile_model(model, device_name="CPU")
    output_layer = compiled_model.output(0)

    metric = evaluate.load("glue", "mrpc")
    for batch in dataset:
        inputs = [numpy.expand_dims(numpy.asarray(batch[key], dtype=numpy.int64), 0) for key in INPUT_NAMES]
        outputs = compiled_model(inputs)[output_layer]
        predictions = outputs[0].argmax(axis=-1)
        metric.add_batch(predictions=[predictions], references=[batch["labels"]])
    metrics = metric.compute()
    f1_score = metrics["f1"]

    return f1_score

print("Checking the accuracy of the original model:")
metric = validate(model, data_source)
print(f"F1 score: {metric:.4f}")

print("Checking the accuracy of the quantized model:")
metric = validate(quantized_model, data_source)
print(f"F1 score: {metric:.4f}")

# Compare Performance of the Original, Converted and Quantized Models
# Compare the original PyTorch model with OpenVINO converted and quantized models (`FP32`, `INT8`) to see the difference in performance. It is expressed in Sentences Per Second (SPS) measure, which is the same as Frames Per Second (FPS) for images.

# Compile the model for a specific device.
compiled_model = core.compile_model(model=model, device_name="CPU")

num_samples = 50
sample = data_source[0]
inputs = {k: torch.unsqueeze(torch.tensor(sample[k]), 0) for k in ["input_ids", "token_type_ids", "attention_mask"]}

with torch.no_grad():
    start = time.perf_counter()
    for _ in range(num_samples):
        torch_model(torch.vstack(list(inputs.values())))
    end = time.perf_counter()
    time_torch = end - start
print(f"PyTorch model on CPU: {time_torch / num_samples:.3f} seconds per sentence, " f"SPS: {num_samples / time_torch:.2f}")

start = time.perf_counter()
for _ in range(num_samples):
    compiled_model(inputs)
end = time.perf_counter()
time_ir = end - start
print(f"IR FP32 model in OpenVINO Runtime/CPU: {time_ir / num_samples:.3f} " f"seconds per sentence, SPS: {num_samples / time_ir:.2f}")

start = time.perf_counter()
for _ in range(num_samples):
    compiled_quantized_model(inputs)
end = time.perf_counter()
time_ir = end - start
print(f"OpenVINO IR INT8 model in OpenVINO Runtime/CPU: {time_ir / num_samples:.3f} " f"seconds per sentence, SPS: {num_samples / time_ir:.2f}")

""" Finally, measure the inference performance of OpenVINO `FP32` and `INT8` models. For this purpose, use [Benchmark Tool](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html) in OpenVINO.

**Note**: The `benchmark_app` tool is able to measure the performance of the OpenVINO Intermediate Representation (OpenVINO IR) models only. For more accurate performance, run `benchmark_app` in a terminal/command prompt after closing other applications. Run `benchmark_app -m model.xml -d CPU` to benchmark async inference on CPU for one minute. Change `CPU` to `GPU` to benchmark on GPU. Run `benchmark_app --help` to see an overview of all command-line options.
 """

print("You are ready to evaluate the results using benchmark_app.")
