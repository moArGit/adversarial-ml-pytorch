# Adversarial Attacks and Defenses on Neural Networks

Adversarial attacks on neural networks have garnered significant attention due to their ability to deceive models with subtle, often imperceptible perturbations. Understanding these attacks and implementing effective defenses is crucial for developing robust AI systems. This project explores various adversarial attacks and defenses, specifically focusing on the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), along with techniques like adversarial training and defensive distillation.

## Table of Contents
- [Overview](#overview)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The primary objectives are to:

- Investigate the impact of adversarial attacks on model performance.
- Implement and evaluate defense mechanisms to enhance model robustness.
- Provide insights into the effectiveness of these defenses against various attack strategies.

### Key Components

1. **Adversarial Attacks**:
   - **Fast Gradient Sign Method (FGSM)**: A single-step attack that generates adversarial examples by applying perturbations based on the gradient of the loss function.
   - **Projected Gradient Descent (PGD)**: An iterative attack that refines adversarial examples over multiple steps, making it more potent than FGSM.

2. **Defense Mechanisms**:
   - **Adversarial Training**: Involves training the model on both clean and adversarial examples to improve its resilience.
   - **Defensive Distillation**: A technique that uses a softened output from a pre-trained model to train a new model, enhancing its robustness against attacks.

## Usage

### Load and Preprocess the MNIST Dataset

The MNIST dataset is loaded and transformed for training and testing. The data is normalized to improve model performance.

### Define and Train the Neural Network Model

A simple Convolutional Neural Network (CNN) is defined with the following architecture:

- **Convolutional Layers**: Two convolutional layers followed by ReLU activations and max pooling.
- **Fully Connected Layers**: Two fully connected layers with dropout for regularization.

### Implement Adversarial Attacks

1. **FGSM Attack**:
   - Generates adversarial examples and evaluates the model's accuracy under attack.
   - Example code snippet:
     ```python
     perturbed_data = fgsm_attack(model, criterion, images, labels, epsilon)
     ```

2. **PGD Attack**:
   - Similar to FGSM but applies iterative perturbations to create more challenging adversarial examples.
   - Example code snippet:
     ```python
     perturbed_data = pgd_attack(model, criterion, images, labels, epsilon, alpha, iters)
     ```

### Implement Defense Mechanisms

1. **Adversarial Training**:
   - The training loop is modified to include adversarial examples, improving the model's robustness.
   - Example code snippet:
     ```python
     train_adversarial(model, train_loader, optimizer, criterion, device, epsilon)
     ```

2. **Defensive Distillation**:
   - A distilled model is trained using softmax outputs from the original model to enhance resistance against attacks.
   - Example code snippet:
     ```python
     targets = softmax_with_temperature(teacher_outputs, temperature)
     ```

### Evaluate Model Robustness

The model's performance is evaluated on clean and adversarial examples, providing insights into its robustness.

## Results

The following table summarizes the model's performance under different conditions:

| **Condition**                     | **Test Accuracy** |
|-----------------------------------|-------------------|
| Clean Test Data                   | 98.99%            |
| FGSM Attack (ε=0.1)              | 89.04%            |
| PGD Attack (ε=0.3)               | 45.18%            |
| After Adversarial Training        | 99.29%            |
| Distilled Model Test Accuracy     | 98.97%            |

### Observations

- The model achieves high accuracy on clean data, demonstrating its effectiveness in digit classification.
- Adversarial attacks significantly reduce accuracy, highlighting the vulnerability of neural networks.
- Implementing adversarial training and defensive distillation effectively mitigates the impact of these attacks, resulting in improved robustness.

## Conclusion

This project demonstrates the critical need for robust AI systems capable of withstanding adversarial attacks. The findings indicate that while adversarial attacks can significantly degrade model performance, effective defense strategies such as adversarial training and defensive distillation can enhance resilience. Future work may explore additional attack methods and defense techniques, as well as the application of these concepts to more complex datasets and models.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
