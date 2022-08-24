# PCBEP: A Point Cloud-based Deep Learning Framework for B-cell Epitope Prediction
# Abstract
The ability to identify B-cell epitopes is specifically important in the field of vaccine design, which is particularly relevant for rapid pandemic response. In the past, a range of computational tools have been developed for predicting B-cell pitopes based on the 3D structure of antigen. However, they have presented limited performance and generalization. Here, we proposed PCBEP, a novel B-cell epitope prediction framework based on the three-dimensional point clouds. It employs multiple Point Transfomer layers to learn the embedding describing the atomic properties, geometric features and evolutionary profile. The built-in self-attention mechanism is able to advance the informative components and message passing operator can model the interactions among neighboring atoms at different scales for stimulating the micro-environmental information for predicting epitopes.
# Installation
```shell
git clone  https://github.com/yuyang-0825/PCBEP && cd PCBEP
conda create --name PCBEP-env python=3.7 -y && conda activate PCBEP-env
pip install -r requirements.txt --upgrade
```
# Running the code
