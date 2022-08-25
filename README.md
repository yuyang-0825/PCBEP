# PCBEP: A Point Cloud-based Deep Learning Framework for B-cell Epitope Prediction

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

## Evaluate our model
First unzip the file:'Data/data_feature_surface.7z'

If you want to evaluate the results of our data.
```shell
python pcbep_test.py
```
If you want to train our 5-fold cross-validation model.
```shell
python pcbep_5fold_train.py
```
## Test your own data 
If you want to use our model to test your own data. Please refer to the following steps:
### 1. Get the epitope label of the data.   
Use 0 1 to encode epitope labels and organize them into fasta format, for example: 
```
>6WIR-C
00000000000000000000000000000000010000000000111100000000000000000000000000000000000111111
>5W2B-A
0000001001100111111000100100000001011111000000000000000000000000000000000000000000000100000011001
>5TFW-O
0000000000000000000000000000000000000000000000000000010100000010000010000000000000000000000000000000000000011110011011001000000000000000000000000000000000
```

### 2. Get the surface label of the data.  
Use 0 1 to encode surface labels and organize them into fasta format, for example:
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

### 5. Get the feature file for your own data
```
python Pretreatment/generate.py -l ../Data/label.txt -s ../Data/surface.txt -p ../Data/data -m ../Data/PSSM -o ../Data/data_feature_surface.txt
```
- `-l`or`--label` file path for epitope label.  [default:'../Data/label.txt']
- `-s`or`--surface` file path for surface label. [default:'../Data/surface.txt']
- `-p`or`--pdb` fold path for pdb files to be tested.  [default:'../Data/data']
- `-m`or`--pssm` fold path for PSSM files of antigens. [default:'../Data/PSSM']
- `-o`or`--output` output file path. [default:'../Data/data_feature_surface.txt']

### 6. Get test result
```
python pcbep_test.py -i Data/data_feature_surface.txt -c checkpoint.pt -o result/result_pcbep.txt
```
- `-i`or`--input` file path for fature path. [default:'Data/data_feature_surface.txt']
- `-c`or`--checkpoint` file path for pre-trained model.. [default:'checkpoint.pt']
- `-o`or`--ouptup` output file path. [default:'result/result_pcbep.txt']

