## DL assignment: Rank-N-Contrast


### Loss Function
#### CV task
The loss function [`RnCLoss`](./loss.py#L34) used for the CV part in [`loss.py`](./loss.py) takes `features` and `labels` as input, return the loss value, and has three hyper-parameters: `temperature`, `label_diff`, and `feature_sim` associated with it.
```python
from loss import RnCLoss

# define loss function with temperature, label difference measure, 
# and feature similarity measure
criterion = RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')

# features: [bs, 2, feat_dim]
features = ...
# labels: [bs, label_dim]
labels = ...

# compute RnC loss
loss = criterion(features, labels)
```

### Running

#### CV task
Download AgeDB dataset from [here](https://ibug.doc.ic.ac.uk/resources/agedb/) and extract the zip file (you may need to contact the authors of AgeDB dataset for the zip password) to folder `./CNN/data`.

- Firstly, change to the CNN folder
    ```
    cd ./CNN
    ```
- To train the model with the L1 loss, run 
    ```
    python main_l1.py
    ```
- To train the model with the RnC framework, first run 
    ```
    python main_rnc.py
    ```
    
    to train the encoder. The checkpoint of the encoder will be saved to `./save`. Then, run
    ```
    python main_linear.py --ckpt <PATH_TO_THE_TRAINED_ENCODER_CHECKPOINT>
    ```
  to train the regressor.

