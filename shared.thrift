# Define shared types for the PA1 Thrift service
typedef list<list<double>> Matrix

struct MLModel {
    1: Matrix V
    2: Matrix W
}

struct MLGradient {
    1: Matrix dV
    2: Matrix dW
}

enum TaskStatus {
    ACCEPTED = 1,
    REJECTED = 2
}

struct TrainingResult {
    1: MLGradient gradient
    2: double error_rate
}