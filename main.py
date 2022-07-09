from graph_and_stats import *
# Need to turn-on GPU runtime for faster training, using CUDA.


if __name__ == '__main__':
    # run all models and creates summary table and graphs
    names = ["LeNet5", "LeNet5_Weight_Decay", "LeNet5_BN", "LeNet5_Dropout"]
    WeightDecay = 0.0001
    LearningRate = 0.001
    Epochs = 20
    BatchSize = 32

    #
    # traing section - comment/uncomment models to train
    #
    builed_and_train(model_name="LeNet5", weight_decay_en=0, learning_rate=LearningRate, epochs=Epochs,
                     batch_size=BatchSize)
    builed_and_train(model_name="LeNet5_Weight_Decay", weight_decay_en=1, weight_decay=WeightDecay,
                     learning_rate=LearningRate, epochs=Epochs, batch_size=BatchSize)
    builed_and_train(model_name="LeNet5_BN", weight_decay_en=0, learning_rate=LearningRate, epochs=Epochs,
                     batch_size=BatchSize)
    builed_and_train(model_name="LeNet5_Dropout", weight_decay_en=0, learning_rate=LearningRate, epochs=Epochs,
                     batch_size=BatchSize)

    # plot all results
    for name in names:
        accuracy_plot(name)
    # printing summary table
    summary_table()

    # an example how to use pre-trained network, choosing names[0..3]
    model_name = names[0]  # LeNet5/LeNet5_Weight_Decay/LeNet5_BN/LeNet5_Dropout
    model = load_model(model_name)
    trained_model_accuracy = test_model(model)
    print(f"The {model_name} trained model test accuracy is: {trained_model_accuracy}")




