# CSNE: Conditional Signed Network Embedding

This repository contains the source code, installation and use instructions for the method presented in the paper: 
*CSNE: Conditional Signed Network Embedding*. Instructions for replicating 
the experiments in the paper are also given.

We provide Python implementations of the complete CSNE model as well as of the MaxEnt priors described in the 
manuscript. The repository is maintained by Alexandru Mara (alexandru.mara(at)ugent.be).

## Installation

Install directly from GitHub with:

```shell
$ pip install git+https://github.com/aida-ugent/CSNE.git
```

Install in development mode with:

```shell
$ git clone https://github.com/aida-ugent/CSNE
$ cd CSNE
$ pip install -e .
```

Where `-e` means "editable" mode.

## Usage

### Running CSNE from the Comand Line
The `csne` command line tool is automatically installed. It parses the command line arguments and runs the model
with takes the following parameters:

```text
  -h, --help            show this help message and exit
  --inputgraph [INPUTGRAPH]
                        Input graph path
  --output [OUTPUT]     Path where the embeddings will be stored.
  --tr_e [TR_E]         Path of the input train edges. Default None (in this
                        case returns embeddings)
  --tr_pred [TR_PRED]   Path where the train predictions will be stored.
                        Default tr_pred.csv
  --te_e [TE_E]         Path of the input test edges. Default None.
  --te_pred [TE_PRED]   Path where the test predictions will be stored.
                        Default te_pred.csv
  --prior_tricount PRIOR_TRICOUNT
                        Toogles triangle count use in prior. (1) use triangles
                        and node polarity, (0) only node polarity. Default is
                        1.
  --prior_learning_rate PRIOR_LEARNING_RATE
                        Learning rate for prior. Default is 1.0.
  --prior_epochs PRIOR_EPOCHS
                        Training epochs for prior. Default is 100.
  --prior_tol PRIOR_TOL
                        Early stop prior fit if grad norm is below this value.
                        Default is 0.0001.
  --prior_regval PRIOR_REGVAL
                        Regularization value, reduces the certainty about 1s
                        and -1s. Default is 0.9
  --use_csne USE_CSNE   Toogle CSNE use. (1) use CSNE, (0) use MaxEnt prior
                        only. Default is 1.
  --learning_rate LEARNING_RATE
                        Learning rate for CSNE. Default is 0.1.
  --epochs EPOCHS       Training epochs for CSNE. Default is 500.
  --s1 S1               Sigma 1. Default is 1.
  --s2 S2               Sigma 2. Default is 2.
  --dimension DIMENSION
                        Dimensionality of the CSNE embeddings. Default is 2.
  --delimiter DELIMITER
                        Delimiter used in the input files.
  --directed            If specified, network treated as directed. Default is
                        undirected.
  --verbose             Determines the verbosity level of the output.

```

**NOTE:** The inputgraph expected contains, in each line: `src,dst,sign` where src and dst are the source and 
destination nodes linked and sign is -1 or +1. If one desires to directly compute predictions for edge pairs, 
these can be provided in the tr_e or te_e parameters. The expected format of these files is, per line: `src,dst`.

Examples of running CSNE:
```bash
# Example 1: Run prior only and compute predictions for edge pairs
csne --inputgraph ./graph.edgelist --tr_e ./tr.edgelist --te_e ./te.edgelist --tr_pred './tr.out' --te_pred './te.out' --use_csne 0
# Example 2: Run full CSNE and store embeddings
csne --inputgraph ./graph.edgelist --output './embeddings.txt'
```

## Reproducing Experiments
In order to reproduce the CSNE results in the paper the following steps are necessary: 

1. Install CSNE dependencies as shown in the [Installation](#Installation) section above
2. Download the datasets: 

    * [Slashdot(a)](https://snap.stanford.edu/data/soc-sign-Slashdot081106.html)
    * [Slashdot(b)](https://snap.stanford.edu/data/soc-sign-Slashdot090216.html)
    * [Epinions](https://snap.stanford.edu/data/soc-sign-epinions.html)
    * [Wiki-rfa](https://snap.stanford.edu/data/wiki-RfA.html)
    * [Bitcoin-alpha](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)
    * [Bitcoin-otc](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)

3. Modify the `.ini` configuration files provided to ensure that the *dataset* paths and *method* paths
match the directories where they have been stored/installed on your machine. Then run the evaluation as:

    ```bash
    python -m evalne ./conf_sign.ini
    ```

**NOTE:** The baseline methods are available here: [SiNE](https://faculty.ist.psu.edu/szw494/codes/SiNE.zip),
[SIGNet](https://github.com/raihan2108/signet/blob/master/signet.py),
[L-SNE/N-SNE](https://github.com/wzsong17/Signed-Network-Embedding). These methods do not accept command-line 
arguments out-of-the-box, so in order to evaluate them using the conf files provided, appropriate mains must 
be created for each method. These main files must take the same parameters shown in the conf files for each method.

## Citation ##

If you have used CSNE in your research, please cite our 
[CIKM2020 paper](https://dl.acm.org/doi/10.1145/3340531.3411959) (video presentation also available in the link):

```
@inproceedings{Mara2020csne,
    author = {Mara, Alexandru and Mashayekhi, Yoosof and Lijffijt, Jefrey and de Bie, Tijl},
    title = {CSNE: Conditional Signed Network Embedding},
    year = {2020},
    isbn = {9781450368599},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3340531.3411959},
    doi = {10.1145/3340531.3411959},
    booktitle = {Proceedings of the 29th ACM International Conference on Information & Knowledge Management},
    pages = {1105–1114},
    numpages = {10},
    location = {Virtual Event, Ireland},
    series = {CIKM '20}
}
```
