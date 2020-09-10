*(in this fork, I try to correct errors in the original repo)*

# Lipschitz Estimation

Repository for the paper [Lipschitz Regularity of deep neural networks:
analysis and efficient estimation](https://arxiv.org/abs/1805.10965)

Basic Python dependencies needed: PyTorch >= 0.3


### Code organisation

* `lipschitz_approximations.py`: many estimators
* `lipschitz_utils.py`: toolbox for the different estimators
* `seqlip.py`: SeqLip and GreedySeqLip
* `training.py`: general scheme for train/test
* `utils.py`: utility functions

### How to run the examples

* to run the MNIST example:
```bash
$ python -m models.mnist_5            # train the network
$ python -m experiments.mnist_get_sv  # save the singular values
$ python -m experiments.mnist         # run the example
```
