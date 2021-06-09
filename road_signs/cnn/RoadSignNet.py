from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Flatten


class RoadSignNet(Module):
    def __init__(self, depth=3, classes=43):
        super(RoadSignNet, self).__init__()
        kernel_size = (3, 3)

        self.cnn_layers = Sequential(
            # First set of (CONV => RELU => CONV => RELU) * 2 => POOL
            Conv2d(depth, 8, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            ReLU(),
            BatchNorm2d(8),
            MaxPool2d(kernel_size=2),
            Conv2d(8, 8, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            ReLU(),
            BatchNorm2d(8),
            MaxPool2d(kernel_size=2),

            # Second set of (CONV => RELU => CONV => RELU) * 2 => POOL
            Conv2d(8, 16, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            ReLU(),
            BatchNorm2d(16),
            Conv2d(16, 16, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            ReLU(),
            BatchNorm2d(16),
            MaxPool2d(kernel_size=2),

            # Third set of (CONV => RELU => CONV => RELU) * 2 => POOL
            Conv2d(16, 32, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            ReLU(),
            BatchNorm2d(32),
            Conv2d(32, 32, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            ReLU(),
            BatchNorm2d(32),
            MaxPool2d(kernel_size=2),

            # # Fourth set of (CONV => RELU => CONV => RELU) * 2 => POOL
            # Conv2d(32, 64, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            # ReLU(),
            # BatchNorm2d(64),
            # Conv2d(64, 64, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            # ReLU(),
            # BatchNorm2d(64),
            # MaxPool2d(kernel_size=2),
            #
            # # Fifth set of (CONV => RELU => CONV => RELU) * 2 => POOL
            # Conv2d(64, 128, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            # ReLU(),
            # BatchNorm2d(128),
            # Conv2d(128, 128, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            # ReLU(),
            # BatchNorm2d(128),
            # MaxPool2d(kernel_size=2),
        )

        self.linear_layers = Sequential(
            # first set of FC => RELU layers
            # Flatten(),
            Linear(2048, 128),
            ReLU(),
            # BatchNorm2d(1),
            # Dropout(0.5),


            # second set of FC => RELU layers
            # Flatten(),
            Linear(128, 128),
            ReLU(),
            # BatchNorm2d(1),
            # Dropout(0.5),

            # softmax classifier
            Linear(128, classes),

            # Remove softmax due to low performance
            # Softmax(dim=1)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
