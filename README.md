# PCBEP: A Point Cloud-based Deep Learning Framework for B-cell Epitope Prediction
## Abstract
The ability to identify B-cell epitopes is specifically important in the field of vaccine design, which is particularly relevant for rapid pandemic response. In the past, a range of computational tools have been developed for predicting B-cell pitopes based on the 3D structure of antigen. However, they have presented limited performance and generalization. Here, we proposed `PCBEP`, a novel B-cell epitope prediction framework based on the three-dimensional point clouds. It employs multiple Point Transfomer layers to learn the embedding describing the atomic properties, geometric features and evolutionary profile. The built-in self-attention mechanism is able to advance the informative components and message passing operator can model the interactions among neighboring atoms at different scales for stimulating the micro-environmental information for predicting epitopes.
## Installation Guide
### Install PCBEP from GitHub
```shell
git clone  https://github.com/yuyang-0825/PCBEP && cd PCBEP
```
### Install dependency packages
1. Install `PyTorch` following the [official guide](https://pytorch.org/get-started/locally/).
1. Install `torch-geometric` following the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
1. Install other dependencies:
```
pip install -r requirements.txt
```
The above steps take 20-25 mins to install all dependencies.
### Python Dependencies
`PCBEP` mainly depends on the Python (3.7+) scientific stack.
```
absl-py==0.13.0
addict==2.4.0
appdirs==1.4.4
astunparse==1.6.3
attrs==21.4.0
backports.zoneinfo==0.2.1
cached-property==1.5.2
cachetools==5.2.0
certifi==2022.5.18.1
charset-normalizer==2.1.0
cycler==0.11.0
decorator==4.4.2
dm-tree==0.1.7
dunamai==1.11.1
etils==0.7.1
flatbuffers==2.0
fonttools==4.33.3
gast==0.4.0
get_version==3.5.4
grpcio==1.47.0
h5py==3.6.0
humanfriendly==10.0
idna==3.3
iniconfig==1.1.1
jax==0.3.16
jaxlib==0.3.15
Jinja2==3.1.1
joblib==1.1.0
kiwisolver==1.4.2
legacy-api-wrap==1.2
leidenalg==0.8.7
libclang==14.0.6
llvmlite==0.36.0
matplotlib==3.3.4
mkl-fft==1.3.1
mkl-random==1.2.2 
mkl-service==2.4.0
munkres==1.1.4
natsort==8.1.0
networkx==2.5.1
numba==0.53.1
numexpr==2.8.1
numpy==1.21.6
oauthlib==3.2.0
packaging==21.3
pandas==1.2.3
patsy==0.5.2
Pillow==8.3.1
pluggy==1.0.0
progress==1.6
protobuf==3.19.4
py==1.11.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pynndescent==0.5.6
pyparsing==3.0.8
pyquaternion==0.9.9
pytest==7.1.2
python-dateutil==2.8.2
python-igraph==0.9.6
pytz==2022.1
pytz-deprecation-shim==0.1.0.post0
PyWavelets==1.3.0
PyYAML==6.0
requests==2.28.1
requests-oauthlib==1.3.1
rpy2==3.1.0
rsa==4.9
scanpy==1.7.2
scikit-image==0.18.1
scikit-learn==0.24.2
scipy==1.6.2
seaborn==0.11.1
session-info==1.0.0
simplegeneric==0.8.1
sinfo==0.3.4
six==1.15.0
statsmodels==0.12.2
stdlib-list==0.8.0
tables==3.7.0
tabulate==0.8.10
termcolor==1.1.0
terminaltables==3.1.0
texttable==1.6.4
threadpoolctl==3.1.0
tifffile==2021.11.2
tomli==2.0.1
toolz==0.12.0
torch==1.7.0
torch-cluster==1.5.9 
torch-geometric==2.0.4
torch-scatter==2.0.5
torch-sparse==0.6.8 
torch-spline-conv==1.2.0 
torchaudio==0.5.1
torchvision==0.6.1+cu101
tqdm==4.64.0
typing==3.7.4.3
typing_extensions==4.2.0
tzdata==2022.1
tzlocal==4.2
umap==0.1.1
umap-learn==0.5.1
urllib3==1.26.9
websocket-client==1.3.3
Werkzeug==2.2.2
wrapt==1.14.1
xlrd==1.2.0
yapf==0.32.0
zipp==3.8.0
```
## Tests
If you want to test the results of our data.
```shell
python pcbep_test.py
```
If you want to train our 5-fold cross-validation results.
```shell
python pcbep_5fold_train.py
```
## Running the code
If you want to train your own data on our tool.
### 1. Get the label of the data.   
1 is an epitope, 0 is a non-epitope. Named "label.txt" in "Data". For example: 
```
>6WIR-C
00000000000000000000000000000000010000000000111100000000000000000000000000000000000111111
>5W2B-A
0000001001100111111000100100000001011111000000000000000000000000000000000000000000000100000011001
>5TFW-O
0000000000000000000000000000000000000000000000000000010100000010000010000000000000000000000000000000000000011110011011001000000000000000000000000000000000
```

### 2. Get the surface of the data.  
1 is surface, 0 is non-surface. Named "surface.txt" in "Data". For example:
```
>6WIR-C
011111111111111111111111111111111110011111111111111111111111111111111111111111111111111111
>5W2B-A
01111111111111111111110111111111111111111111111111111111111111111111111111111111011111111111011111
>5TFW-O
01111111111111111111101110011011110011101110111111111011111110010111111111100000101111101111111111101000010111111111011111111111110110111111101110111111111
```

### 3. Get the all pdb file after extracting the single chain.  
 Named "6WIR-C.pdb" in "Data/data". For example
```
ATOM   3334  N   SER C  41      68.159  -1.843 -11.815  1.00122.88           N  
ATOM   3335  CA  SER C  41      67.329  -3.021 -12.051  1.00122.82           C  
ATOM   3336  C   SER C  41      65.929  -2.841 -11.479  1.00127.97           C  
```

### 4. Get the all PSSM matrices file after extracting the single chain.  
 Named "6wir-C.pssm" in "Data/PSSM". For example
```
Last position-specific scoring matrix computed, weighted observed percentages rounded down, information per position, and relative weight of gapless real matches to pseudocounts
            A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    1 S     2  -1   0  -1  -1   0  -1   0  -1  -2  -2  -1  -1  -3  -1   4   2  -3  -2  -1   26   0   0   0   0   0   0   0   0   0   0   0   0   0   0  65   9   0   0   0  0.40 0.03
    2 D    -1  -2   1   5  -3   0   2  -1  -1  -3  -4  -1  -3  -4  -2   2   0  -4  -3  -3    0   0   0  68   0   0   8   0   0   0   0   0   0   0   0  23   0   0   0   0  0.71 0.04
    3 Y    -1  -2  -2  -3  -2  -2  -2  -3   0   0   0  -2   0   3  -3  -1   2   0   5   1    0   0   0   0   0   0   0   0   0   0   0   0   0  16   0   0  22   0  39  23  0.36 0.03  
```

### 5. Get your feature file 
 It named "data_feature_surface.txt" in "Data".
```
python Pretreatment/generate.py
```
- `-l`or`--label`to set the label file path. 
  - default:'../Data/label.txt'
- `-s`or`--surface`to set the surface file path. 
  - default:'../Data/surface.txt'
- `-p`or`--pdb`to set the pdb folder path. 
  - default:'../Data/data'
- `-m`or`--pssm`to set the pssm folder path. 
  - default:'../Data/PSSM'
- `-o`or`--output`to set the output file path. 
  - default:'../Data/data_feature_surface.txt'
- e.g.
```
python Pretreatment/generate.py -l ../Data/label.txt -p ../Data/data -o ../Data/data_feature_surface.txt
```

### 6. Get the result
```
python pcbep_test.py
```
- `-i`or`--input`to set the imput file path. 
  - default:'Data/data_feature_surface.txt'
- `-c`or`--checkpoint`to set the checkpoint file path. 
  - default:'checkpoint.pt'
- `-o`or`--ouptup`to set the output file path. 
  - default:'result/result_pcbep.txt'
- e.g.
```
python pcbep_test.py -i Data/data_feature_surface.txt
```
