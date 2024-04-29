## DL assignment: Rank-N-Contrast

#### NOTE : Python version 3.9.x is recommended.

### Loss Function
The loss function [`RnCLoss`](./CNN/loss.py#L34) used for the CV part in [`loss.py`](./CNN/loss.py) takes `features` and `labels` as input, return the loss value, and has three hyper-parameters: `temperature`, `label_diff`, and `feature_sim` associated with it.

```python
from loss import RnCLoss

# define loss function with temperature, label difference measure, and feature similarity measure
# we have kept the following config constant for all the experiments conducted:

criterion = RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')
loss = criterion(features, labels) # features: (bs, 2, fear_dim), labels: (bs, label_dim)
```

### Running the code for the reproducibility section
Firstly, download the [AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/) dataset and extract the zip file (we contacted the authors of the AgeDB dataset for the zip password) to folder `./CNN/data`.

Change to the CNN folder, then run the file as needed. `main_l1.py` trains the model with just L1 loss in a 1-stage setting. `main_rnc.py` trains the encoder with the RnC framework and saves it in `./save`. Finally, run `main_linear.py`, which uses the trained encoder to train the regressor on top in a 2-stage setting.
```
cd ./CNN

python main_l1.py

python main_rnc.py

python main_linear.py --ckpt <your_checkpoint_from_rnc>
```


### Apart from the official reproduction, we have also tried out this loss on a Graph Regression Task, the ESOL dataset. <br>
It is publicly available as a subset of MoleculeNet <a href="https://moleculenet.org/datasets-1">here</a>.

The detailed results are available <a href="https://docs.google.com/spreadsheets/d/1HBiUqcsvInXPTq7ywI10QNTG04TnSSEAMmGR4vD-u18/edit#gid=1568203868">here</a> with run plots on my wandb account <a href="https://wandb.ai/karannb/dl-project/table?nw=nwuserkarannb">here</a>. A brief summary is below - 
| Method / Loss | test MAE | test RMSE | test MSE | validation MAE | validation RMSE | validation MSE |
| :-----------: | :------: | :-------: | :------: | :------------: | :-------------: | :------------: |
| normal-L1 | 0.247	| 0.326 | 0.106 | **0.224** | 0.325 | 0.106 |
| RnC(L1) + freeze | **0.219** | **0.297** | 0.088 | 0.255 | 0.359 | 0.129 |
| RnC(L1) | **0.212** | 0.314 | 0.099 | 0.235 | 0.345 | 0.119 |
| RnC(L2) | 0.266 | 0.342 | 0.117 | 0.276 | 0.366 | 0.134 |
| RnC(Huber) | 0.242 | 0.326 | 0.106 | 0.245 | **0.317** | **0.101** |

A (<a href="https://umap-learn.readthedocs.io/en/latest/">UMAP</a>) view on the amazing representation space learned by RnC v/s end-to-end L1, <br>
Training Data (**left**), Test Data (**right**)<br>
<img src="imgs/representation_space_train.png" alt="train data" width="400" height="200"/>
<img src="imgs/representation_space_test.png" alt="test data" width="400" height="200"/>

A sequence of commands to reporduce our results - (can be done completely on the free version of Google CoLab, though you might run into python version issues, some known issues are listed below)<br>
- First create a directory called "data/" and create a subfolder titled "ESOL", then you can create the preprocessed dataset as
    ```python
    python3 -m GNN.preprocess
    ```

- The exact runs can be reproduced by running the 3 scripts, they have already been updated with optimal parameters as the default ones. (Note that, we have done some hyperparameter tuning but not a very rigorous grid search)
    ```python
    python3 -m GNN.main_l1
    python3 -m GNN.main_rnc
    python3 -m GNN.main_linear --ckpt <your_checkpoint_from_rnc> [OPTINAL] --freeze_encoder --loss [l1(default)/MSE/huber]
    ```

- The plots for the representation space can be reproduced using 
    ```python
    python3 -m misc.representation_space
    ```
<br>
Everything will be by default logged to w&b and so I recommend signing in using (and pasting your API Key)

```bash
wandb login
```
Following is a bash script to run on multiple GPUs or slurm clusters - 

```bash
#!/bin/sh

#SBATCH --job-name=GNN
#SBATCH --output=logs/l1_GNN.out
#SBATCH --error=logs/l1_GNN.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=128G

export CUDA_VISIBLE_DEVICES=2

cd GNN/

python3 main_l1.py --data_folder data/

# other possible runs are
# python3 main_l1.py --data_folder data/, with output=logs/L1_GNN.out and error=logs/L1_GNN.err
# python3 main_rnc.py --data_folder data/, with output=logs/rnc_GNN.out and error=logs/rnc_GNN.err

# python3 main_linear.py --data_folder data/ --loss [L1(default)/MSE/huber] --ckpt <path> [OPTIONAL] --freeze_encoder, 
# with output=logs/linear_GNN.out and error=logs/linear_GNN.err
```
#### Known issues - 
1. You might not be able to download all PyG dependencies, on a higher python version (> 3.9.19), especially the ones that have a ~cu117~ at the end, **all** experiments can be reproduced without those libraries so feel free to remove them!

### Thanks
- [kaiwenzha/Rank-N-Contrast](https://github.com/kaiwenzha/Rank-N-Contrast): Repository code helped in the reproducibility section.
- [AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/): Dataset used for CV task in the reproducibility section.
- [ESOL](https://pubs.acs.org/doi/10.1021/ci034243x): Dataset used for GNN task in the extension on the reproduction.
