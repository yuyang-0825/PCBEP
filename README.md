# PCBEP: A Point Cloud-based Deep Learning Framework for B-cell Epitope Prediction
## Abstract
The ability to identify B-cell epitopes is specifically important in the field of vaccine design, which is particularly relevant for rapid pandemic response. In the past, a range of computational tools have been developed for predicting B-cell pitopes based on the 3D structure of antigen. However, they have presented limited performance and generalization. Here, we proposed PCBEP, a novel B-cell epitope prediction framework based on the three-dimensional point clouds. It employs multiple Point Transfomer layers to learn the embedding describing the atomic properties, geometric features and evolutionary profile. The built-in self-attention mechanism is able to advance the informative components and message passing operator can model the interactions among neighboring atoms at different scales for stimulating the micro-environmental information for predicting epitopes.
## Installation
```shell
git clone https://github.com/yuyang-0825/PCBEP && cd PCBEP
conda create --name PCBEP-env python=3.7 -y && conda activate PCBEP-env
pip install -r requirements.txt --upgrade
```
## Tests
Redhat、Fedora、Centos:
```shell
yum install p7zip
cd Data
7z x data_feature_surface.7z
cd ..
```
Debian、Ubuntu:
```shell
apt-get install p7zip
cd Data
7z x data_feature_surface.7z
cd ..
```

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

### 6. Get the result
```
python pcbep_test.py
```
