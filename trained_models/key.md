# Model Information

The neural network configuration is in the `neural_network.py` file.

## Key

The file names of the models follow the following convention:

`Xc-Xl-Xp-Xe-cnn_model-#X.pth`

Where `X` is a number.

* `Xc` is the number of convolutional layers used in the neural network.
* `Xl` is the number of linear layers used in the neural network.
* `Xp` is the max pooling and stride numbers in the neural network.
* `Xe` is the number of epochs the neural network was trained for.
* The number after the `#` shows the chronological order in which the models were trained.

### Example

`2c-3l-2p-3e-cnn_model-#2.pth`
is a neural network with **$2$ convolutional layers**,
**$3$ linear layers**,
**max pooling over a $(2, 2)$** window,
**trained for $3$ epochs**,
and was the **$2^{\text {nd}}$ model to be trained**.
