![Python](https://img.shields.io/badge/python-3.10-blue)
![NumPy](https://img.shields.io/badge/numpy-%23C53030)
![Status](https://img.shields.io/badge/status-completed-success)

# Artificial Neural Networks from Scratch

A Python implementation of a **multilayer artificial neural network (ANN)** built from scratch without using high-level machine-learning frameworks.

This repository demonstrates the fundamentals of ANN architecture, training via backpropagation, and visual representation of the network structure.

---

## Overview

This project implements a simple feed-forward neural network in Python using **NumPy** for numerical operations and **matplotlib** for visualization.

The script allows you to:

- Define a custom hidden layer size  
- Train the network on a small dataset  
- Visualize the network architecture and connections  
- Observe output behavior after training

This project reflects a foundational understanding of neural network mechanics, especially backpropagation and layered computations.

---

## Motivation

Artificial neural networks are foundational machine learning models that approximate non-linear relationships in data by learning weights and biases through iterative training processes. This implementation aims to reinforce core deep learning concepts by building a neural network *without* relying on high-level libraries such as TensorFlow or PyTorch. :contentReference[oaicite:0]{index=0}

---

## Key Features

- Fully implemented multilayer ANN  
- Custom hidden layer configuration  
- Training using backpropagation  
- Visualization of network neurons and connections  
- Standalone Python implementation (no ML frameworks required)

---

## Technologies Used

- **Python** – Implementation language  
- **NumPy** – Numerical computation  
- **matplotlib** – Visualization  
- **Git & GitHub** – Version control

---

## Repository Structure

Artificial-Neural-Networks/
├── ArtificialNeuralNetwork.py # Neural network training & visualization
└── README.md # Documentation


---

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/beyzaekrem/Artificial-Neural-Networks.git
Navigate to the project folder:

cd Artificial-Neural-Networks
Create a virtual environment (optional but recommended):

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
Install the required packages (NumPy and matplotlib):

pip install numpy matplotlib
Running the Neural Network
To train and visualize the network:

Run the Python script:

python ArtificialNeuralNetwork.py
When prompted, enter the number of neurons for the hidden layer.

The model trains using a small dataset and prints error metrics at intervals.

A network graphic window displays the final neural architecture connections.

What This Project Demonstrates
This repository highlights:

Manual implementation of neural network forward and backward propagation

Weight and bias updates using gradient descent

Layered network design principles

Visualization of network topology for interpretability

Future Improvements
This project can be extended with:

Larger datasets and dynamic data loaders

Multi-class classification support

GUI interface for interactive parameter tuning

Exporting learned weights for reuse

Integration with educational notebooks

Author
Developed by Beyza Ekrem
Computer Engineering Student
