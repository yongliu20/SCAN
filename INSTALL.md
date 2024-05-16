## Installation

### Requirements
- Linux with Python ≥ 3.8
- PyTorch ≥ 1.10 and torchvision that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

### Usage

Install required packages. 

```bash
conda create -n scan python=3.8
conda activate scan
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
pip install -r requirements.txt
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```



Install other packages.

```bash
cd scan/modeling/pixel_decoder/ops
sh make.sh
```