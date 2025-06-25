"""
what dose model_builder.py do? -> this is where we keep all the model building logic for the Going Modular project.
the architecture is based TinyVGG
"""

import torch
from torch import nn


class TinyVGG(nn.Module):
    """
    Creats the Tiny Vgg Arcitecture
    
    Args:
        input_shape: int
        hidden_units: int - number of hidden units in the model
        output_shape: int - number of output classes
    """
    def __init__(self, input_shape:int , hidden_units:int, output_shape:int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*56*56, out_features=output_shape)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

if __name__ == "__main__":
    # test the model
    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=3)
    print(model)
    
    # create a dummy input tensor
    x = torch.randn(size=(32, 3, 224, 224))
    
    # pass the dummy input through the model
    output = model(x)
    
    # print the output shape
    print(f"Output shape: {output.shape}")