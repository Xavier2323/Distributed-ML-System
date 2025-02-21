import glob
import sys
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('/home/hsu00191/Distributed_Systems/thrift-0.19.0/lib/py/build/lib*')[0])

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
            print(f"[DEBUG] Before update - Shared Gradient sum: {np.sum(self.gradient)}")
            self.gradient = sum_matricies(self.gradient, local_gradient)
            print(f"[DEBUG] After update - Shared Gradient sum: {np.sum(self.gradient)}")

    def average(self, num_jobs):
        """ Compute average gradient """
        with self.lock:
            return scale_matricies(self.gradient, 1 / max(num_jobs, 1))  # Prevent division by zero

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
        if self.scheduling_policy == 1:  # Random scheduling
            return random.choice(self.compute_nodes)
        else:  # Load-balanced scheduling
            return min(self.compute_nodes, key=lambda node: random.random())

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
        self.mlp_model.init_training_random(f"{dir}/train_letters1.txt", k, h)
        V, W = self.mlp_model.get_weights()

        # Shared gradient containers
        shared_gradient_V = SharedGradient(V.shape)
        shared_gradient_W = SharedGradient(W.shape)

        for r in range(rounds):
            print(f"[TRAINING ROUND {r+1}/{rounds}]")

            # Reset gradients for the round
            shared_gradient_V.gradient = np.zeros_like(shared_gradient_V.gradient)
            shared_gradient_W.gradient = np.zeros_like(shared_gradient_W.gradient)

            threads = []
            
            work_queue = [f"{dir}/train_letters{i}.txt" for i in range(1, 12)]  # Assume 10 training files

            # Spawn threads for each training file
            for training_file in work_queue:
                node_host, node_port = self._select_compute_node()
                t = threading.Thread(
                    target=self.thread_func, 
                    args=(node_host, node_port, training_file, shared_gradient_V, shared_gradient_W, V, W, eta, epochs)
                )
                threads.append(t)
                t.start()

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Average gradients and update weights
            avg_gradient_V = shared_gradient_V.average(len(work_queue))
            avg_gradient_W = shared_gradient_W.average(len(work_queue))

            if avg_gradient_W.shape == W.shape and avg_gradient_V.shape == V.shape:
                self.mlp_model.update_weights(avg_gradient_V, avg_gradient_W)
                print(f"[DEBUG] Weights Updated - W sum: {np.sum(self.mlp_model.W)}, V sum: {np.sum(self.mlp_model.V)}")
            else:
                print("[ERROR] Gradient shapes do not match. Skipping update.")


            # Validate model
            val_error = self.mlp_model.validate(f"{dir}/validate_letters.txt")
            print(f"[VALIDATION ERROR] After round {r+1}: {val_error:.4f}")

        return val_error

    def thread_func(self, node_host, node_port, training_file, shared_gradient_V, shared_gradient_W, V, W, eta, epochs):
        """ Worker thread for training a single batch """
        try:
            # Connect to compute node
            transport = TSocket.TSocket(node_host, node_port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            client = ComputeNode.Client(protocol)
            transport.open()

            # Initialize training on compute node
            model = MLModel(W=W.tolist(), V=V.tolist())
            status = client.initializeTraining(training_file, model)

            if status == TaskStatus.ACCEPTED:
                result = client.trainModel(eta, epochs)
                local_gradient_V = np.array(result.gradient.dV)
                local_gradient_W = np.array(result.gradient.dW)

                # Update shared gradients
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

    # Initialize coordinator with compute node list
    handler = CoordinatorHandler(scheduling_policy, "compute_nodes.txt")
    processor = Coordinator.Processor(handler)
    transport = TSocket.TServerSocket(port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    # Start the server
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    print(f"[STARTED] Coordinator listening on port {port} with scheduling policy {scheduling_policy}")
    server.serve()

if __name__ == "__main__":
    main()