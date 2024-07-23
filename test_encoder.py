import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, name, n=1):
        super(ConvolutionBlock, self).__init__()
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.poolings = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(n):
            conv = nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.convs.append(conv)
            self.activations.append(nn.ReLU())
            self.batch_norms.append(nn.BatchNorm2d(out_channels))
            self.poolings.append(nn.MaxPool2d(kernel_size=2))
            self.dropouts.append(nn.Dropout2d(p=dropout))

    def forward(self, x):
        for conv, activation, batch_norm, pooling, dropout in zip(self.convs, self.activations, self.batch_norms, self.poolings, self.dropouts):
            x = conv(x)
            x = activation(x)
            x = batch_norm(x)
            x = pooling(x)
            x = dropout(x)
        return x

class EncoderBlockSeparated(nn.Module):
    def __init__(self, in_channels, filters=1, dropout=0, n_convs=1, n_blocks=3, name=''):
        super(EncoderBlockSeparated, self).__init__()
        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            in_ch = in_channels if i == 0 else (filters * 2) // (i)
            out_ch = (filters * 2) // (i + 1)
            block = ConvolutionBlock(in_ch, out_ch, dropout, f'{name}_block{i+1}', n=n_convs)
            self.blocks.append(block)

        self.flatten = nn.Flatten()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.flatten(x)
        return x


# Example test function to verify the network
def test_network():
    # Instantiate the model
    model = EncoderBlockSeparated(in_channels=3, filters=16, dropout=0.3, n_convs=2, n_blocks=3, name='encoder')
    
    # Create a sample input tensor with batch size 1, 3 channels, and 64x64 image size
    sample_input = torch.randn(1, 3, 64, 64)
    
    # Perform a forward pass through the model
    output = model(sample_input)
    
    # Print the output shape
    print(f'Output shape: {output.shape}')

# Run the test function
test_network()

