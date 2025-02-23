include "shared.thrift"

service ComputeNode {
    shared.TaskStatus initializeTraining(1: string filename, 2: shared.MLModel model),
    shared.TrainingResult trainModel(1: double eta, 2: i32 epochs)
}

service Coordinator {
    double train(1: string dir, 2: i32 rounds, 3: i32 epochs, 4: i32 h, 5: i32 k, 6: double eta),
    
    ## Additional helper functions
    # list<string> getAvailableNodes(),
    # double getModelAccuracy()
}