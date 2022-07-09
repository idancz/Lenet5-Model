from torch import nn

# Basic LeNet5 model with capability to change activation function and select between MaxPool2d/AvgPool2d


class LeNet5(nn.Module):
    def __init__(self, activation="ReLu", polling="Max"):
        super().__init__()
        # Model layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1,
                               padding=2)  # in: 32x32x1, out: 28x28x6
        # self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)  # in: 14x14x6 , out: 10x10x16
        # self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.polling = nn.MaxPool2d(kernel_size=2, stride=2)
        if (polling == "Avg"):
            self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)  # in: 5x5x16, out: 400x120
        self.flatten = nn.Flatten(start_dim=1)  # in: 400x120, out:1x400*120
        self.fc1 = nn.Linear(in_features=120, out_features=84)  # in: 1x400*120 , out=1x120*84
        self.fc2 = nn.Linear(in_features=84, out_features=10)  # in:1x120*84 , out=1x84*10

        self.activation = nn.ReLU()
        if (activation == "Tanh"):
            self.activation = nn.Tanh()
        print(f"\n\nLeNet5 with activation={activation} and polling={polling}\n")

    # forward
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.polling(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.polling(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


# Inherited leNet5 class with tunable Dropouts layers
class LeNet5Dropout(LeNet5):
    def __init__(self, activation="ReLu", polling="Max", dropout_ratio=0.3):
        super(LeNet5Dropout, self).__init__(activation, polling)
        self.dropout = nn.Dropout(dropout_ratio)
        print(f"LeNet5Dropout with activation={activation} and polling={polling}\n")

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.polling(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.polling(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.dropout(x)  # adding droupout layer1
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)  # adding droupout layer2
        x = self.fc2(x)
        return x


# Inherited leNet5 class with 2 Batch normalization layers
class LeNet5BatchNorm(LeNet5):
    def __init__(self, activation="ReLu", polling="Max"):
        super(LeNet5BatchNorm, self).__init__(activation, polling)
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm1d(84)
        print(f"LeNet5BatchNorm with activation={activation} and polling={polling}\n")

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.polling(x)
        x = self.batch_norm1(x)  # adding batch normalization layer1
        x = self.conv2(x)
        x = self.activation(x)
        x = self.polling(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.batch_norm2(x)  # adding batch normalization layer2
        x = self.activation(x)
        x = self.fc2(x)
        return x

