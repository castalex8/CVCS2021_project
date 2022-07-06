from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Flatten


class RoadSignNet(Module):
    def __init__(self, classes: int = 1, depth: int = 3, is_retrieval: bool = False):
        super(RoadSignNet, self).__init__()
        kernel_size = (3, 3)
        classes = 2 if is_retrieval else classes

        self.cnn_layers = Sequential(
            # First set of (CONV => RELU => CONV => RELU) * 2 => POOL
            Conv2d(depth, 8, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            ReLU(),
            BatchNorm2d(8),
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

            # Third set of (CONV => RELU => CONV => RELU) * 2 => POOL
            # Conv2d(32, 64, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            # ReLU(),
            # BatchNorm2d(64),
            # Conv2d(64, 64, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            # ReLU(),
            # BatchNorm2d(64),
            # MaxPool2d(kernel_size=2),
            #
            # # Third set of (CONV => RELU => CONV => RELU) * 2 => POOL
            # Conv2d(64, 128, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            # ReLU(),
            # BatchNorm2d(128),
            # Conv2d(128, 128, kernel_size=kernel_size, padding_mode='zeros', padding=kernel_size),
            # ReLU(),
            # BatchNorm2d(128),
            # MaxPool2d(kernel_size=2),
        )

        self.flatten = Flatten()
        self.linear_layers = Sequential(
            # first set of FC => RELU layers
            # Linear(8192, 128),
            Linear(3872, 128),
            ReLU(),

            # second set of FC => RELU layers
            Linear(128, 128),
            ReLU(),

            # softmax classifier
            Linear(128, classes),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x
