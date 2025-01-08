# Systematic and Scalable Metamorphic Testing for Vision-Based Autonomous Driving

This repository provides the implementation of the **Systematic and Scalable Metamorphic Testing Framework for Vision-Based Autonomous Driving Systems (ADS)**. The framework integrates **causal inference** and **diffusion models** to enhance the reliability testing of ADS, facilitating the detection of faults under diverse driving conditions.

## NOTE:
**The code is currently undergoing necessary refinement and will be available soon.**

## Folder Structure

- `Autopilot/`: Contains 5 implementations of the tested autonomous driving algorithms and the original test images that need to be mutated

- `CausalEffect/`: Structural causal models, counterfactual models, and a multi-objective search framework for critical test conditions.

- `Metamorphic/`: Apply the test conditions found in the semantic description, the low-risk test images are generalized to the high-risk driving conditions based on the fine-tuned diffusion model.

- `TACTIC/`: Representative Metamorphic Testing baselines for comparison.
  - `do_testing/`: Main Implementations to perform TACTIC test
  - `logger/`: TACTIC search records
  - `munit/`: Contains random MUNIT method and DeepRoad method
  - `test_outputs/`: Contains the mutation results of the baselines
  - `train_outputs/`: Neuron coverage during training of the algorithm under test
- `requirements.txt`: A list of Python dependencies required to run the project.