import glob
import sys
sys.path.insert(0, glob.glob('/home/hsu00191/Distributed_Systems/thrift-0.19.0/lib/py/build/lib*')[0])
sys.path.append('gen-py')

import time
import numpy as np
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from service import ComputeNode
from shared.ttypes import MLModel, TaskStatus

def test_compute_node(dataset_file):
    """ Tests a single compute node by initializing and training a small dataset. """

    # Connect to the compute node
    transport = TSocket.TSocket("localhost", 9091)  # Assuming running on port 9091
    transport = TTransport.TBufferedTransport(transport)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = ComputeNode.Client(protocol)

    transport.open()

    # Initialize random weights
    np.random.seed(42)
    h, k, d = 20, 26, 16  # 16 input features, 20 hidden, 26 output classes
    V = (np.random.rand(h+1, k) * 0.02) - 0.01
    W = (np.random.rand(d+1, h) * 0.02) - 0.01
    model = MLModel(V=V.tolist(), W=W.tolist())

    # Initialize the training
    status = client.initializeTraining(dataset_file, model)
    if status == TaskStatus.REJECTED:
        print("[FAILED] Compute node rejected training initialization")
        transport.close()
        return

    print("[SUCCESS] Compute node initialized training")

    # Train the model for multiple epochs
    eta = 0.001  # Learning rate
    epochs = 50  # Train for 50 epochs
    training_result = client.trainModel(eta, epochs)

    if training_result.error_rate < 0:
        print("[FAILED] Training returned an invalid error rate")
    else:
        print(f"[SUCCESS] Training completed with error rate: {training_result.error_rate}")

    transport.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_compute_node.py dataset_file")
        sys.exit(1)
    dataset_file = str(sys.argv[1])
    test_compute_node(dataset_file)
