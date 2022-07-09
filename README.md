# Lenet5-Model
Implementation of Lenet5 Model on the Fashion-MNIST dataset 

# Program Description
## Background
Comparing the result of Lenet5 model on the Fashion-MNIST dataset using the following techniques:  
- Dropout (at the hidden layer)
- Weight Decay 
- Batch normalizaion

### Additional Info
1. Two Dropout layers 30% 
2. Batch Normalization layers size of 84 and 10
3. Weight Decay with alpha = 0.0001
4. The Basic Lenet object is made of 3 conv2d layers of 5x5, 2 down-sampler 2x2, 2xFC layers , 4 ReLU activation functions after conv layers
5. The following parameters were used
6. WeightDecay = 0.0001
7. LearningRate = 0.001
8. Epochs = 20
9. BatchSize = 32

## Training
Train using the following:  
1. names = ["LeNet5", "LeNet5_Weight_Decay", "LeNet5_BN", "LeNet5_Dropout"]
2. comment/uncomment one of the following functions:
    -	builed_and_train(model_name="LeNet5", weight_decay_en=0, learning_rate=LearningRate, epochs=Epochs, batch_size=BatchSize)
    - builed_and_train(model_name="LeNet5_Weight_Decay", weight_decay_en=1, weight_decay=WeightDecay, learning_rate=LearningRate, epochs=Epochs, batch_size=BatchSize)
    - builed_and_train(model_name="LeNet5_BN", weight_decay_en=0, learning_rate=LearningRate, epochs=Epochs, batch_size=BatchSize)
    - builed_and_train(model_name="LeNet5_Dropout", weight_decay_en=0, learning_rate=LearningRate, epochs=Epochs, batch_size=BatchSize)

## Validation
1. Choose model name from  names list.  
an example : use pre-trained network, by choosing names[0..3] and pass it to test_model function<br />
3. example:  
model_name = names[0]  # choose from  LeNet5/LeNet5_Weight_Decay/LeNet5_BN/LeNet5_Dropout<br />
model = load_model(model_name)<br />
trained_model_accuracy = test_model(model)<br />
print(f"The {model_name} trained model test accuracy is: {trained_model_accuracy}")<br />

## Results
I found that using Weight Decay and dropout has shown best performance while batch normalization showed lower test accuracy.<br />
As we expected the weight Decay solved issues of vanishing gradient hence better performance was observed.<br />
Batch normalization should have normalized the data staples but we believe due to the small batch-size (32) it did not affect much.<br />
I consider also adding more batch normalization layers (e.g. after each CONV and FC layers ) that may improve, or to play with the normalization factor.
<br />
![image](https://user-images.githubusercontent.com/108329249/178121733-b6dff568-908b-47ee-9bad-0c7650268d8e.png)

<br />

## Graphs
### LeNet5 basic model (90.55%)
![image](https://user-images.githubusercontent.com/108329249/178121751-b508c0d7-b8eb-4977-9823-a99cfdab9e81.png)


### LeNet5 basic model  with weight decay of 0.0001 (91.04%)

![image](https://user-images.githubusercontent.com/108329249/178121762-833143ce-181f-4cad-98c3-084f88ef0e1b.png)


### LeNet5 basic model with Batch normalization of 6 and 84 (90.29%,epch 8)
![image](https://user-images.githubusercontent.com/108329249/178121773-688a5746-077a-40fb-8a88-39ef4d3de462.png)


### LeNet5 basic model with dropout rate of 0.001  (90.86%, epoch 12)

![image](https://user-images.githubusercontent.com/108329249/178121797-7d58992d-862d-4083-8cb1-8ccbb60176d9.png)













