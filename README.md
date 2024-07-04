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

## Download the model



python ./utils/download_glue_data.py --data_dir='glue_data' --tasks='MRPC'

