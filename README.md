1. Overview
    This project implements a Reinforcement Learning based Virtual Network Embedding with Alternative Paths (VNEAP) system. The agent learns to select the best alternative virtual network request and map its nodes onto a physical substrate network. It uses Maskable PPO from sb3-contrib to handle invalid action masking during both the alternative selection phase and the node mapping phase. Two heuristic baselines (Greedy and TANTO proxy) are included for comparison. The implementation is in a single Python script.

2. Files
    rl_orchestrator.py: Main script that defines the substrate and request generators, the Gymnasium environment, heuristic agents, the custom feature extractor, training loop, evaluation logic, and plotting.

3. Requirements

    a. Dependencies:
        i. Python 3
        ii. PyTorch
        iii. gymnasium
        iv. numpy
        v. networkx
        vi. stable-baselines3
        vii. sb3-contrib
        viii. matplotlib

    b. To install: pip install torch gymnasium numpy networkx stable-baselines3 sb3-contrib matplotlib

4. Running on Google Colab
    a. Open Google Colab at https://colab.research.google.com
    b. Create a new notebook
    c. In the first cell, install the required packages:
        !pip install torch gymnasium numpy networkx stable-baselines3 sb3-contrib matplotlib
    d. Upload vne_ap.py to the Colab environment or paste the code into a cell
    e. Run the script using: !python rl_orchestrator.py
    f. Alternatively, copy the code directly into notebook cells and run them sequentially
    
5. Environment Design
    a. Substrate Network:
        i. 15 nodes generated using a Barabasi-Albert graph with 3 preferential attachments
        ii. Each node has a random CPU capacity between 50 and 100
        iii. Each link has a random bandwidth capacity between 50 and 100

    b. Virtual Network Requests:
        i. Each request has 2 alternatives generated as path graphs with 3 to 4 nodes
        ii. Alternative 1 is compute heavy (CPU 20 to 30, BW 5 to 10)
        iii. Alternative 2 is bandwidth heavy (CPU 5 to 10, BW 20 to 30)

    c. Two-Phase Action Space:
        i. Phase 1 (Selection): Agent picks which alternative to embed (action 0 or 1)
        ii. Phase 2 (Mapping): Agent maps each virtual node to a substrate node one at a time
        iii. Invalid actions are masked so the agent can only choose feasible substrate nodes

    d. Reward Structure:
        i. Successful full embedding: +50
        ii. Failed mapping due to CPU: -50
        iii. Failed mapping due to bandwidth: -20
        iv. Per-node mapping cost: -0.05 times the CPU demand

    e. Departure Process:
        i. Active deployments are released with a 15 percent probability per step to free resources

6. Heuristic Baselines
    a. Greedy Agent:
        i. Selects the alternative with the lowest total resource cost (sum of CPU and BW demands)
        ii. Maps nodes using first-fit on substrate nodes

    b. TANTO Proxy Agent:
        i. Evaluates current CPU and bandwidth utilization of the substrate
        ii. Selects the alternative that minimizes demand on the more stressed resource
        iii. Maps nodes using first-fit on substrate nodes
7. RL Agent
    a. Algorithm: Maskable PPO from sb3-contrib
    b. Policy: MultiInputPolicy with a custom feature extractor
    c. Feature Extractor:
        i. Encodes substrate CPU values through a linear layer (15 to 32 units, ReLU)
        ii. Encodes flattened request matrix through a linear layer (8 to 32 units, ReLU)
        iii. Concatenates both embeddings and projects to 64-dimensional output
    d. Training: 25000 timesteps by default

8. How to Run
    a. Run the script: python vne_ap.py
    b. The script will:
        i. Train the RL agent for 25000 timesteps
        ii. Evaluate all three agents (RL, Greedy, TANTO) on 200 target requests each
        iii. Generate four plots comparing performance

9. Output
    a. Plot A: Bar chart of accepted requests per agent out of 200 attempts
    b. Plot B: Bar chart of rejection rate percentage per agent
    c. Plot C: Line plots of substrate CPU and bandwidth utilization over simulation steps
    d. Plot D: Smoothed RL training reward curve over timesteps
    e. Training logs are saved to the ./tmp/ directory
