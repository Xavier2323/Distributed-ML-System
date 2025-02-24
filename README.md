# Distributed Systems: Project 1

### Jun-Ting Hsu (hsu00191@umn.edu), Thomas Knickerbocker (knick073@umn.edu)


## Running Our Code:

Results after 10 epochs w/ client command python3 client.py localhost 8090 ./ML/letters 30 10 20 0.0001, a 0.0 load probability amongst 4 nodes, and a coordinator scheduling_policy of 1:
[Phase 1 Full Run](./assets/phase1_30_epochs_ss.png) <br>

Results after 5 epochs w/ client command python3 client.py localhost 8090 localhost 9090 ./ML/letters 50 75 20 0.0001, a 0.0 load probability amongst 4 nodes, and a coordinator scheduling_policy of 1:
[Phase 1 Partial Run (more intensive)](./assets/phase1_5_epochs_ss.png) <br>


In Client Terminal: python3 client.py localhost 9090 ./ML/letters 50 75 20 0.0001

In Coordinator Terminal: python3 coordinator.py 9090 1

In Node Terminals: python3 compute_node.py <portNo> 0.0
^where portNo is 9091, 9092, 9093, and/or 9094

Results after 5 epochs w/ client params localhost 9090 ./ML/letters 50 75 20 0.0001:

