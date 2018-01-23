# A Deep-Reinforcement Learning Approach for Software-Defined Networking Routing Optimization

###### [1709.07080](https://arxiv.org/abs/1709.07080): Giorgio Stampa, Marta Arias, David Sanchez-Charles, Victor Muntes-Mulero, Albert Cabellos

In this paper we design and evaluate a Deep-Reinforcement Learning agent that optimizes routing. Our agent adapts automatically to current traffic conditions and proposes tailored configurations that attempt to minimize the network delay. Experiments show very promising performance. Moreover, this approach provides important operational advantages with respect to traditional optimization algorithms.

---

Code and datasets [here](https://github.com/knowledgedefinednetworking/a-deep-rl-approach-for-sdn-routing-optimization/releases).

---

# Keras and Deep Deterministic Policy Gradient to control an OMNeT++ network simulator

## How to build?

1. Use a modern Linux or macOS system.
1. Install OMNeT++ version 4.6 in your system (please see instructions at http://omnetpp.org).
1. Run 'make' from inside 'omnet/router'. This will generate `networkRL` which is needed by the python script.
1. Install Python 3.6 in your system.
1. Install the packages listed in `requirements.txt` (please use *exact* versions). Virtualenv could be of help (https://pypi.python.org/pypi/virtualenv).


## How to train?

### Single run with fixed parameters
Reads configuration from ```DDPG.json```

```
python3 ddpg.py
```

## How to play?

### Single run with fixed parameters
Reads configuration (ddpg, neural network weights, etc.) from ```folder```

```
python3 ddpg.py play folder
```

---

### EXAMPLE JSON CONFIG

```
{
    "ACTIVE_NODES": 3,                                          # number of active nodes in the network
    "ACTUM": "NEW",                                             # action: NEW or DELTA
    "BATCH_SIZE": 50,                                           # size of learning batch
    "BUFFER_SIZE": 2000,                                        # max size of replay buffer
    "ENV": "label",                                             # "label" or "balancing"
    "EPISODE_COUNT": 10,                                        # number of episodes
    "EXPLORE": 0.8,                                             # exploration: rate if <=1, number of steps otherwise
    "GAMMA": 0.99,                                              # discount factor
    "HACTI": "selu",                                            # non-linear activation function for hidden layers
    "HIDDEN1_UNITS": 300,                                       # neurons of layer 1
    "HIDDEN2_UNITS": 600,                                       # neurons of layer 2
    "LRA": 0.0001,                                              # learning rate of the actor network
    "LRC": 0.001,                                               # learning rate of the critic network
    "MAX_STEPS": 1000,                                          # number of steps per episode
    "MU": 0.0,                                                  # Ornstein-Uhlenbeck process' μ
    "PRAEMIUM": "MAX",                                          # reward function
    "PRINT": true,                                              # verbosity
    "ROUTING": "Linkweight",                                    # "Balancer", "Linkweight", "Pathchoice"
    "RSEED": null,                                              # random seed: null or number
    "SIGMA": 0.07,                                              # Ornstein-Uhlenbeck process' σ
    "STATUM": "T",                                              # state representation: T or RT
    "TAU": 0.001,                                               # soft target update
    "THETA": 0.03,                                              # Ornstein-Uhlenbeck process' θ
    "TRAFFIC": "EXP"                                            # traffic: static or changing (~randomly)
}
```

---

author: giorgio@ac.upc.edu

* [Keras](https://keras.io/)
* [DDPG](https://arxiv.org/abs/1509.02971)
* [OMNeT++](https://omnetpp.org/)
