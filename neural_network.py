from torch import nn

# Neural network configurations
input_layer = 100 * 100
layer_1 = 64
layer_2 = 128
layer_3 = 256
output_layer = 21


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_layer, layer_1),
            nn.ReLU(),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, layer_3),
            nn.ReLU(),
            nn.Linear(layer_3, output_layer)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
