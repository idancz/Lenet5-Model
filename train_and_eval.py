import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torchvision
from torchvision import transforms

from lenet5 import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_accuracy_dict = {}
train_accuracy_dict = {}
path = "models/"

# Trainer Class which gets training parameters : model, optimizer, criterion, num epochs, batch-size , train data sets and test datasets.

class Trainer:
    def __init__(self, model, optimizer, criterion, epochs, batch_size, train_dataset, test_dataset):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    # train function of class Trainer.
    def train(self):
        self.model.train()
        total_loss, avg_loss, accuracy, nof_samples, correct_labeled_samples = 0, 0, 0, 0, 0

        train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            prediction = self.model(inputs)
            loss = self.criterion(prediction, targets.long())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            correct_labeled_samples += (prediction.argmax(1) == targets).sum().item()
            nof_samples += len(targets)
            accuracy = (correct_labeled_samples / nof_samples) * 100

        return tuple([avg_loss, accuracy])

    # test function for class testing
    def test(self):
        self.model.eval()
        total_loss, avg_loss, accuracy, nof_samples, correct_labeled_samples = 0, 0, 0, 0, 0

        test_dataloader = DataLoader(self.test_dataset, self.batch_size, shuffle=False)
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                prediction = self.model(inputs)
                loss = self.criterion(prediction, targets.long())

                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                correct_labeled_samples += (prediction.argmax(1) == targets).sum().item()
                nof_samples += len(targets)
                accuracy = (correct_labeled_samples / nof_samples) * 100

        return tuple([avg_loss, accuracy])

    # run function for training and testing
    def run(self, model_name="LeNet5"):
        global train_accuracy_dict
        global test_accuracy_dict
        global path
        best_acc = 0
        train_accuracy_dict[model_name] = []
        test_accuracy_dict[model_name] = []
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()
            train_accuracy_dict[model_name].append(train_acc)
            test_accuracy_dict[model_name].append(test_acc)
            print(
                f'epoch {epoch} | train loss : {train_loss:.3f} | train accuracy: {train_acc:.2f} % | test loss: {test_loss:.3f} | test accuracy: {test_acc:.2f} %')
            if (test_acc > best_acc):
                torch.save(self.model.state_dict(), path + model_name + ".pt")
                best_acc = test_acc


# loading pre trained model , geting model_name as parameter.
def load_model(model_name):
    global path
    models = {  # case select using anonmous function lambda
        "LeNet5": lambda: LeNet5(),
        "LeNet5_Weight_Decay": lambda: LeNet5(),
        "LeNet5_Dropout": lambda: LeNet5Dropout(dropout_ratio=0.3),
        "LeNet5_BN": lambda: LeNet5BatchNorm()
    }
    model = models[model_name]().to(device)
    print(f'Loading {path + model_name + ".pt"}')
    model.load_state_dict(torch.load(path + model_name + ".pt"))
    model.eval()  # evaluating only
    return model


# main calling function or constructing and training
def builed_and_train(pre_trained_cont=False, model_name="LeNet5", weight_decay_en=0, weight_decay=0.0001,
                     learning_rate=0.001, epochs=20, batch_size=32):
    global train_accuracy_dict
    global test_accuracy_dict
    global path
    # Load dataset
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),  # randomly flip and rotate
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(1, 1)  # (mean=0.3814, std=0.3994)
    ])
    train_set = torchvision.datasets.FashionMNIST(root='./data',
                                                  train=True,
                                                  download=True,
                                                  transform=transform)

    test_set = torchvision.datasets.FashionMNIST(root='./data',
                                                 train=False,
                                                 download=True,
                                                 transform=transform)

    models = {
        "LeNet5": lambda: LeNet5(),
        "LeNet5_Weight_Decay": lambda: LeNet5(),
        "LeNet5_Dropout": lambda: LeNet5Dropout(dropout_ratio=0.3),
        "LeNet5_BN": lambda: LeNet5BatchNorm()
    }

    model = models[model_name]().to(device)

    if pre_trained_cont:
        model.load_state_dict(torch.load(path + model_name + ".pt"))

    # Load loss function
    criterion = nn.CrossEntropyLoss()

    # Build optimizer
    if weight_decay_en:
        optimizer = optim.Adam(params=model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)  # can be SGD, but Adam works better
    else:
        optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)  # can be SGD but Adam works better

    # Create Trainer with all the parameters
    trainer = Trainer(model, optimizer, criterion, epochs, batch_size, train_set, test_set)
    trainer.run(model_name)


# testing pre-trained model
def test_model(model):
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),  # randomly flip and rotate
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(1, 1)  # (mean=0.3814, std=0.3994)
    ])
    test_set = torchvision.datasets.FashionMNIST(root='./data',
                                                 train=False,
                                                 download=True,
                                                 transform=transform)

    trainer = Trainer(model=model, optimizer=None,
                      criterion=nn.CrossEntropyLoss(),
                      epochs=1,
                      batch_size=16,
                      train_dataset=None,
                      test_dataset=test_set)
    _, test_accuracy = trainer.test()
    return test_accuracy
