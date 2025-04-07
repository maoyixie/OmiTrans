# OmiTrans (Only for NTU course CE7412)

## Getting Started

### Prerequisites
-   CPU or NVIDIA GPU + CUDA CuDNN
-   [Python](https://www.python.org/downloads) 3.6+
-   Python Package Manager
    -   [Anaconda](https://docs.anaconda.com/anaconda/install) 3 (recommended)
    -   or [pip](https://pip.pypa.io/en/stable/installing/) 21.0+
-   Python Packages
    -   [PyTorch](https://pytorch.org/get-started/locally) 1.2+
    -   TensorBoard 1.10+
    -   Tables 3.6+
    -   prefetch-generator 1.0+
-   [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 2.7+

### Installation
-   Clone the repo
```bash
git clone https://github.com/maoyixie/OmiTrans.git
cd OmiTrans
```
-   Install the dependencies
    -   For conda users  
    ```bash
    conda env create -f environment.yml
    conda activate omitrans
    ```
    -   For pip users
    ```bash
    pip install -r requirements.txt
    ```
    
### Try it out
-   Put the gene expression data (A.tsv) and DNA methylation data (B.tsv) in the default data path (./data)
-   Train and test using the default settings
```bash
python train_test.py
```
-   Check the output files
```bash
cd checkpoints/test/
```
-   Visualise the metrics and losses
```bash
tensorboard --logdir=tb_log --bind_all
```
