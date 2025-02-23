import glob
import sys
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../thrift-0.19.0/lib/py/build/lib*')[0])

import random
import threading
import numpy as np
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from service.Coordinator import Iface
from shared.ttypes import TaskStatus, MLModel, TrainingResult
from service import ComputeNode
from service import Coordinator
from ML.ML import mlp, scale_matricies, sum_matricies

# Thread-safe shared gradient storage
class SharedGradient:
    def __init__(self, shape):
        self.gradient = np.zeros(shape)
        self.lock = threading.Lock()

    def update(self, local_gradient):
        with self.lock:
            self.gradient = sum_matricies(self.gradient, local_gradient)

    def average(self, num_jobs):
        with self.lock:
            return scale_matricies(self.gradient, 1 / max(num_jobs, 1))  # Prevent division by zero

    def reset(self):
        with self.lock:
            self.gradient = np.zeros_like(self.gradient)

class CoordinatorHandler(Iface):
    def __init__(self, scheduling_policy, compute_nodes_file):
        self.scheduling_policy = scheduling_policy
        self.mlp_model = mlp()
        self.compute_nodes = self._load_compute_nodes(compute_nodes_file)

    def _load_compute_nodes(self, filename):
        """ Reads compute nodes from file and returns a list of (host, port) tuples """
        nodes = []
        try:
            with open(filename, "r") as file:
                for line in file:
                    host, port = line.strip().split(",")
                    nodes.append((host, int(port)))
            print(f"[INFO] Loaded {len(nodes)} compute nodes.")
        except Exception as e:
            print(f"[ERROR] Failed to load compute nodes from {filename}: {e}")
        return nodes

    def _select_compute_node(self):
        """ Selects a compute node based on the scheduling policy """
        if self.scheduling_policy == 1:  
            return random.choice(self.compute_nodes)  # Random scheduling
        else:  
            return min(self.compute_nodes, key=lambda node: random.random())  # Simple load balancing

    def train(self, dir, rounds, epochs, h, k, eta):
        """
        Runs distributed training over multiple rounds using the compute nodes.
        - dir: Directory containing training/validation data
        - rounds: Number of training rounds
        - epochs: Training epochs per round
        - h, k: Hidden & output layer sizes
        - eta: Learning rate
        """
        # Initialize model with random weights
        success = self.mlp_model.init_training_random(f"{dir}/train_letters1.txt", k, h)
        if not success:
            print("[ERROR] MLP model initialization failed. Check dataset path.")
            return -1
        
        V, W = self.mlp_model.get_weights()
        print(f"[DEBUG] Initial Weights: W shape {W.shape}, V shape {V.shape}")

        shared_gradient_V = SharedGradient(V.shape)
        shared_gradient_W = SharedGradient(W.shape)

        for r in range(rounds):
            print(f"[TRAINING ROUND {r+1}/{rounds}]")

            # Retrieve latest weights and reset gradients
            V, W = self.mlp_model.get_weights()
            shared_gradient_V.reset()
            shared_gradient_W.reset()

            threads = []
            work_queue = [f"{dir}/train_letters{i}.txt" for i in range(1, 12)]  

            for training_file in work_queue:
                node_host, node_port = self._select_compute_node()
                t = threading.Thread(
                    target=self.thread_func, 
                    args=(node_host, node_port, training_file, shared_gradient_V, shared_gradient_W, V, W, eta, epochs)
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
            
            avg_gradient_V = shared_gradient_V.average(len(work_queue))
            avg_gradient_W = shared_gradient_W.average(len(work_queue))

            if avg_gradient_W.shape == W.shape and avg_gradient_V.shape == V.shape:
                scaled_gradient_V = scale_matricies(avg_gradient_V, -eta)
                scaled_gradient_W = scale_matricies(avg_gradient_W, -eta)
                
                # check it's updating the weights
                print(f"[DEBUG] Scaled Gradients: dW sum {np.sum(scaled_gradient_W)}, dV sum {np.sum(scaled_gradient_V)}")
                self.mlp_model.update_weights(scaled_gradient_V, scaled_gradient_W)
                
                print(f"[DEBUG] Updated Weights: W {W[:2]}, V {V[:2]}")
                
            else:
                print("[ERROR] Gradient shapes do not match. Skipping update.")

            # print the prediction
            
            # Validate model after each round
            val_error = self.mlp_model.validate(f"{dir}/train_letters11.txt")
            print(f"[VALIDATION ERROR] After round {r+1}: {val_error:.4f}")

        return val_error

    def thread_func(self, node_host, node_port, training_file, shared_gradient_V, shared_gradient_W, V, W, eta, epochs):
        """ Worker thread for training a single batch """
        try:
            transport = TSocket.TSocket(node_host, node_port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            client = ComputeNode.Client(protocol)
            transport.open()

            model = MLModel(W=W.tolist(), V=V.tolist())
            status = client.initializeTraining(training_file, model)

            if status == TaskStatus.ACCEPTED:
                result = client.trainModel(eta, epochs)
                local_gradient_V = np.array(result.gradient.dV)
                local_gradient_W = np.array(result.gradient.dW)

                print(f"[DEBUG] Received Gradients: dW sum {np.sum(local_gradient_W)}, dV sum {np.sum(local_gradient_V)}")

                shared_gradient_V.update(local_gradient_V)
                shared_gradient_W.update(local_gradient_W)

            transport.close()

        except Exception as e:
            print(f"[ERROR] Compute node {node_host}:{node_port} failed - {e}")

def main():
    """ Main entry point for starting the coordinator server """
    if len(sys.argv) != 3:
        print("Usage: python3 coordinator.py <port> <scheduling_policy>")
        sys.exit(1)

    port = int(sys.argv[1])
    scheduling_policy = int(sys.argv[2])

    handler = CoordinatorHandler(scheduling_policy, "compute_nodes.txt")
    processor = Coordinator.Processor(handler)
    transport = TSocket.TServerSocket(port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    print(f"[STARTED] Coordinator listening on port {port} with scheduling policy {scheduling_policy}")
    server.serve()

if __name__ == "__main__":
    main()
