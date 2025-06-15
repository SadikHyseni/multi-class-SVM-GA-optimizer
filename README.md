# SVM Classifier with Genetic Algorithm Optimization

This project implements a multi-class classification system using a **Support Vector Machine (SVM)**, where the regularization parameter **C** is optimized via a **Genetic Algorithm (GA)**. It also incorporates **Two-Fold Cross-Validation** to ensure robust model evaluation.

---

## Project Summary

- **Goal**: Improve the accuracy of a multi-class SVM classifier using hyperparameter optimization.
- **Technique**: Genetic Algorithm for tuning the SVM's `C` parameter.
- **Evaluation**: Two-Fold Cross-Validation for validation; detailed metrics (precision, recall, F1-score) for final test evaluation.

---

## Key Features

- One-vs-All SVM for multi-class classification.
- Genetic Algorithm to optimize the regularization parameter `C`.
- Custom implementation of:
  - Data standardization
  - SGD-based SVM training
  - Confusion matrix and performance metrics
- Accuracy, precision, recall, and F1-score computed for each class.
- Implemented entirely in **Java** with CSV data handling.

---

## Algorithms Used

- **Support Vector Machine (SVM)** with Stochastic Gradient Descent
- **Genetic Algorithm (GA)**:  
  - Fitness based on cross-validation accuracy  
  - Roulette wheel selection  
  - Crossover (average of parents)  
  - Gaussian mutation  
  - Elitism for best individuals
- **Two-Fold Cross-Validation** for fitness evaluation

---

## Results

- **Optimal C** (regularization parameter) found: `0.01`
- **Validation Accuracy**: `95.3%`
- **Test Accuracy**: `94.05%`
- High performance on digits like 0, 4, and 6.
- Minor confusion between similar digits (e.g., 8 and 9).

##  Dependencies
- **Java 8 or higher**
- **No external libraries required (pure Java)**
- **CSV data files in data/ directory**

