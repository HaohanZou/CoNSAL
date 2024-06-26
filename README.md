# CoNSAL

This repository contains the source code for the Paper: Combining Neural Networks and Symbolic Regression for Analytical Lyapunov Function Discovery.

## Requirement

* [Pytorch:2.0.1](https://pytorch.org/get-started/locally/) 
* [PySR](https://astroautomata.com/PySR/)

## How it works

This algorithm comprises three components: a learner, a symbolic regression model, and a falsifier. The learner employs an Input Convex Neural Network (ICNN) to minimize the Lyapunov risk and identify a neural Lyapunov function. The symbolic regression model approximates the neural network through analytical formulas. The falsifier checks the Lyapunov conditions on the analytical formula in specified state space of the dynamics.


## A typical procedure is as follows:

* Define the parameters for symbolic regression model
* Define a dynamical system
* Set checking conditions in root finding falsifier
* Initialize the neural network with random parameters for neural Lyapunov function training
* Start training and verifying
* Procedure stops when no counterexample is found


The training process iteratively updates the parameters by minimizing the Lyapunov risk, a cost function that quantifies the violation of Lyapunov conditions. During the verification phase, counterexamples are periodically identified and added to the training set for subsequent iterations. This method enhances the neural network's ability to generalize and find a valid neural Lyapunov function.



## Examples
* [Van Der Pol Oscillator](https://github.com/HaohanZou/CoNSAL/tree/main/Linear_Path_Following)

* [Linear Path Following](https://github.com/HaohanZou/CoNSAL/tree/main/Van_Der_Pol_Oscillator)

## Citation
```
@misc{feng2024combiningneuralnetworkssymbolic,
      title={Combining Neural Networks and Symbolic Regression 
      for Analytical Lyapunov Function Discovery}, 
      author={Jie Feng and Haohan Zou and Yuanyuan Shi},
      year={2024},
      eprint={2406.15675},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2406.15675}, 
}
```