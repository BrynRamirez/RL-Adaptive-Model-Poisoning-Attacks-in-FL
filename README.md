# Adaptive Model Poisoning Attacks in Federated Learning Using Reinforcement Learning
## Installation
Requires **python3.9**, make sure you have python3.9 before executing. 
Create a virtual environment before installing the dependencies:
> python3.9 -m venv venv
> 
> source venv/bin/activate

You can install the dependencies with pip3:
>pip3 install -e
## Running the experiment
server.py is where the code ran. To run server.py:

if on GPU cluster (UTRGV's) run simulation.sh 
(you need to create your own virtual environment before running on the cluster)
> sbatch run-simulation.sh
>
if running locally run server.py
> python federated_learning/server.py

server.py can be changed to run simulations or attack strategy at your choice by commenting out sections.

## Location of Data
The location of the Data is located in [Federated Metrics](federated_learning/Federated%20Metrics) this contains all the graphs and
csv's for each simulation.The simulation directory is labeled by year month and day followed by the run ID. For example,
[20250430_184652](federated_learning/Federated%20Metrics/20250430_184652). If you'd like to know
the most run it is at the bottom directory 'federated_learning/Federated Metrics'