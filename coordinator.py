import glob
import sys
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../thrift-0.19.0/lib/py/build/lib*')[0])

import time
import uuid
import logging
import os
from datetime import datetime
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("coordinator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("coordinator")

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
        self.node_load = {node: 0 for node in self.compute_nodes}  # Tracks active jobs per node
        self.lock = threading.Lock()  # Ensure thread safety when modifying `node_load`
        
        # Ensure log directory exists
        os.makedirs("logs", exist_ok=True)
        
    def _select_compute_node(self, attempt=0):
        """ Selects a compute node based on the scheduling policy """
        if self.scheduling_policy == 1:  
            # Select a node randomly according to the scheduling policy
            selected_node = random.choice(self.compute_nodes)
            return selected_node
        else:  
            # Sort nodes by load
            sorted_nodes = sorted(self.node_load.keys(), key=lambda node: self.node_load[node])
            
            # Get the node with the attempt-th lowest load (with wraparound)
            attempt_index = attempt % len(sorted_nodes)
            selected_node = sorted_nodes[attempt_index]
            
            return selected_node

    def _acquire_node(self, node):
        """Attempt to acquire a node and check if it's available"""
        node_host, node_port = node
        try:
            transport = TSocket.TSocket(node_host, node_port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            client = ComputeNode.Client(protocol)
            transport.open()
            
            # Random scheduling: accept immediately
            if self.scheduling_policy==1:
                return client, transport, True
            # If our scheduling policy is 2, we must check if the node should accept the task
            elif self.scheduling_policy==2 and client.should_accept_task():
                self._increment_node_load(node)  # Mark the node as handling a job
                return client, transport, True
            else:
                transport.close()
                return None, None, False
        except Exception as e:
            logger.error(f"Failed to acquire node {node_host}:{node_port}: {str(e)}")
            return None, None, False

    def _increment_node_load(self, node):
        """ Increment the job count for a node """
        # with self.lock:
        self.node_load[node] += 1

    def _decrement_node_load(self, node):
        """ Decrement the job count for a node """
        # with self.lock:
        self.node_load[node] = max(0, self.node_load[node] - 1)

    def thread_func(self, job_id, training_file, shared_gradient_V, shared_gradient_W, V, W, eta, epochs, max_retries=100):
        """ Worker thread for training a single batch """
        attempt = 0
        acquired_node = False
        node = None
        client = None
        transport = None
        
        while attempt < max_retries and not acquired_node:
            with self.lock:  # Lock here to avoid threads all trying to use the same node:
                # Select node according to scheduling policy - pass attempt as parameter
                node = self._select_compute_node(attempt)
                logger.info(f"Attempting to acquire {node}")
                # Attempt to acquire the node before trying to use it
                client, transport, acquired_node = self._acquire_node(node)

            # Wait a small amount of time before retrying
            if not acquired_node:
                attempt += 1
                time.sleep(0.01 * (2 ** attempt))  # Exponential backoff
                
        if not acquired_node:
            logger.error(f"Thread failed to acquire a compute node after {max_retries} attempts")
            return
        
        node_host, node_port = node
        logger.info(f"[{job_id}] starting task on node {node_host}:{node_port} for file {training_file}")
        init_time = time.time()
        
        try:
            model = MLModel(W=W.tolist(), V=V.tolist())
            status = client.initializeTraining(training_file, model)

            if status == TaskStatus.ACCEPTED:
                # Start training time measurement
                train_start = time.time()
                result = client.trainModel(eta, epochs)
                train_time = time.time() - train_start
                
                local_gradient_V = np.array(result.gradient.dV)
                local_gradient_W = np.array(result.gradient.dW)

                shared_gradient_V.update(local_gradient_V)
                shared_gradient_W.update(local_gradient_W)

            elif status == TaskStatus.REJECTED:
                logger.error(f"[{job_id}] Compute node {node_host}:{node_port} failed to initialize the model")

        except Exception as e:
            logger.error(f"[{job_id}] Compute node {node_host}:{node_port} failed - {e}")

        finally:
            # Finish timing, close transport channel, log, & decrement node load
            task_duration = time.time() - init_time
            if transport and transport.isOpen():
                transport.close()
            logger.info(f"[{job_id}] Compute node completed in {task_duration:.3f}s")
            self._decrement_node_load(node)  # Mark the job as complete


    def _load_compute_nodes(self, filename):
        """ Reads compute nodes from file and returns a list of (host, port) tuples """
        nodes = []
        try:
            with open(filename, "r") as file:
                for line in file:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        host, port = parts
                        nodes.append((host, int(port)))
                logger.info(f"[INFO] Loaded {len(nodes)} compute nodes.")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load compute nodes from {filename}: {e}")
        return nodes

    def train(self, dir, rounds, epochs, h, k, eta):
        """
        Runs distributed training over multiple rounds using the compute nodes.
        - dir: Directory containing training/validation data
        - rounds: Number of training rounds
        - epochs: Training epochs per round
        - h, k: Hidden & output layer sizes
        - eta: Learning rate
        """
        job_id = f"{uuid.uuid4().hex[:8]}"
        job_start_time = time.time()
        
        # Create a job-specific log file
        job_log_file = f"logs/job-{job_id}.log"
        job_file_handler = logging.FileHandler(job_log_file)
        job_file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(job_file_handler)
        
        logger.info(f"[{job_id}] Starting new training job with parameters: "
                   f"dir={dir}, rounds={rounds}, epochs={epochs}, h={h}, k={k}, eta={eta}")
        
        # Initialize model with random weights
        success = self.mlp_model.init_training_random(f"{dir}/train_letters1.txt", k, h)
        if not success:
            logger.error("[ERROR] MLP model initialization failed. Check dataset path.")
            return -1
        
        V, W = self.mlp_model.get_weights()
        logger.debug(f"[DEBUG] Initial Weights: W shape {W.shape}, V shape {V.shape}")

        shared_gradient_V = SharedGradient(V.shape)
        shared_gradient_W = SharedGradient(W.shape)

        for r in range(rounds):
            logger.info(f"[TRAINING ROUND {r+1}/{rounds}]")

            # Retrieve latest weights and reset gradients
            V, W = self.mlp_model.get_weights()
            shared_gradient_V.reset()
            shared_gradient_W.reset()

            threads = []
            # Change this back to process all files
            work_queue = [f"{dir}/train_letters{i}.txt" for i in range(1, 12)]

            for training_file in work_queue:
                t = threading.Thread(
                    target=self.thread_func, 
                    args=(job_id, training_file, shared_gradient_V, shared_gradient_W, V, W, eta, epochs)
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
            
            avg_gradient_V = shared_gradient_V.average(len(work_queue))
            avg_gradient_W = shared_gradient_W.average(len(work_queue))

            if avg_gradient_W.shape == W.shape and avg_gradient_V.shape == V.shape:
                
                # check to see if weights are being updated, then update them:
                logger.debug(f"[DEBUG] Avg Absolute Gradients: dW sum {np.sum(np.abs(avg_gradient_W))}, dV sum {np.sum(np.abs(avg_gradient_V))}")
                self.mlp_model.update_weights(avg_gradient_V, avg_gradient_W)
                
                # Verify weights were updated
                new_V, new_W = self.mlp_model.get_weights()
                
            else:
                logger.error("[ERROR] Gradient shapes do not match. Skipping update.")
            
            # Validate model after each round
            val_error = self.mlp_model.validate(f"{dir}/train_letters11.txt")
            logger.error(f"[VALIDATION ERROR] After round {r+1}: {val_error:.4f}")
            
        total_job_time = time.time() - job_start_time
        logger.info(f"Job {job_id} completed in {total_job_time} seconds")
        return val_error

def main():
    """ Main entry point for starting the coordinator server """
    if len(sys.argv) != 3:
        logger.error("Usage: python3 coordinator.py <port> <scheduling_policy>")
        sys.exit(1)

    port = int(sys.argv[1])
    scheduling_policy = int(sys.argv[2])

    handler = CoordinatorHandler(scheduling_policy, "compute_nodes.txt")
    processor = Coordinator.Processor(handler)
    transport = TSocket.TServerSocket(host='0.0.0.0', port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    logger.info(f"[STARTED] Coordinator listening on port {port} with scheduling policy {scheduling_policy}")
    server.serve()

if __name__ == "__main__":
    main()
