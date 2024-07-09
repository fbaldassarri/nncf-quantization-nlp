**Code for blog post**: https://open.substack.com/pub/baremetalai/p/compress-it-8bit-post-training-model

## Prerequisite

You have conda installed (anaconda, or miniconda).

## Prepare the Environment 

```
(base) user@host:~/$ conda create -n nncf-workspace -c intel python=3.11 -y

(base) user@host:~/$ conda activate nncf-workspace

(nncf-workspace) user@host:~/$ python -m pip install --upgrade pip

```

## Install Neural Network Compressor Framework and Intel OpenVINO™

In addition to torch compiled and optmized for cpu usage.

```
(nncf-workspace) user@host:~/$ pip install nncf>=2.5.0

(nncf-workspace) user@host:~/$ pip install torch transformers "torch>=2.1" datasets evaluate tqdm  --extra-index-url https://download.pytorch.org/whl/cpu

(nncf-workspace) user@host:~/$ pip install openvino>=2024.2.0

```
## Run the script

```
(nncf-workspace) user@host:~/$ git clone https://github.com/fbaldassarri/nncf-quantization-nlp

(nncf-workspace) user@host:~/$ cd nncf-quantization-nlp

(nncf-workspace) user@host:~/nncf-quantization-nlp$ python main.py

```

## Benchmarking 

Measure the inference performance of OpenVINO `FP32` and `INT8` models.
For this purpose, use [Benchmark Tool](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html) in OpenVINO to benchmark async inference on CPU for one minute.

```
(nncf-workspace) user@host:~/nncf-quantization-nlp$ benchmark_app -m ./model/bert_mrpc.xml -shape [1,128],[1,128],[1,128] -d CPU -api sync

(nncf-workspace) user@host:~/nncf-quantization-nlp$ benchmark_app -m ./model/quantized_bert_mrpc.xml -shape [1,128],[1,128],[1,128] -d CPU -api sync
```
