import glob
import sys
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('/home/hsu00191/Distributed_Systems/thrift-0.19.0/lib/py/build/lib*')[0])

from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from service import Coordinator

def main():
    """ Client to interact with the coordinator and start training. """

    if len(sys.argv) != 8:
        print("Usage: python3 client.py <coordinator_ip> <coordinator_port> <dir_path> <rounds> <epochs> <h> <eta>")
        sys.exit(1)

    # Parse command-line arguments
    coordinator_ip = sys.argv[1]
    coordinator_port = int(sys.argv[2])
    dataset_dir = sys.argv[3]
    rounds = int(sys.argv[4])
    epochs = int(sys.argv[5])
    h = int(sys.argv[6])  # Number of hidden units
    eta = float(sys.argv[7])  # Learning rate
    k = 26  # Fixed (26 output classes for letter recognition)

    # Connect to the coordinator
    try:
        transport = TSocket.TSocket(coordinator_ip, coordinator_port)
        transport = TTransport.TBufferedTransport(transport)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        client = Coordinator.Client(protocol)
        transport.open()

        print(f"[INFO] Requesting training from Coordinator at {coordinator_ip}:{coordinator_port}")
        print(f"Training Params -> Rounds: {rounds}, Epochs: {epochs}, H: {h}, eta: {eta}")

        # Call train() on the coordinator
        final_validation_error = client.train(dataset_dir, rounds, epochs, h, k, eta)

        print(f"[SUCCESS] Training completed. Final validation error: {final_validation_error:.4f}")

        transport.close()

    except Exception as e:
        print(f"[ERROR] Failed to connect to Coordinator: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()