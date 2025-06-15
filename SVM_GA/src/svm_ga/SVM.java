package svm_ga;

public class SVM {
	private double C;
	private double[] weights; 
	private double bias; 

	private final int MAX_ITERATIONS = 1000; 
	private final double INITIAL_LEARNING_RATE = 0.01;
	private final double LEARNING_RATE_DECAY = 0.99; 
	private final double EARLY_STOPPING_THRESHOLD = 0.001; 

	// Constructor to set the regularization parameter (C)
	public SVM(double C) {
		this.C = C;
	}

	// train the SVM using Stochastic Gradient Descent (SGD)
	public void train(double[][] trainingData) {
		int numSamples = trainingData.length;
		int numFeatures = trainingData[0].length - 1;

		// Initialize the weights and bias
		weights = new double[numFeatures];
		bias = 0;

		double learningRate = INITIAL_LEARNING_RATE;

		// Gradient Descent
		for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
			double batchLoss = 0.0;
			int batchCount = 0;

			for (int i = 0; i < numSamples; i++) {
				// Extract features and label from the dataset
				double[] features = new double[numFeatures];
				for (int j = 0; j < numFeatures; j++) {
					features[j] = trainingData[i][j];
				}
				int label = (int) trainingData[i][numFeatures];

				// Compute prediction using decision function
				double prediction = decisionFunction(features);

				// Update weights and bias using SGD
				if (label * prediction < 1) {
					// Misclassification, update weights and bias
					for (int j = 0; j < numFeatures; j++) {
						weights[j] = weights[j] - learningRate * (2 * C * weights[j] - label * features[j]);
					}
					bias = bias + learningRate * label;
					batchLoss += 1 - label * prediction;
				} else {
					// No misclassification, only update regularization term
					for (int j = 0; j < numFeatures; j++) {
						weights[j] = weights[j] - learningRate * (2 * C * weights[j]);
					}
				}

				batchCount++;

				// Early stopping if the loss improvement is below a threshold
				if (batchCount > 100 && batchLoss < EARLY_STOPPING_THRESHOLD) {
					System.out.println("Early stopping at iteration " + iter);
					return;
				}
			}

			// Apply learning rate decay for each iteration
			learningRate *= LEARNING_RATE_DECAY;
		}
	}

	// Method to predict the class of a new data point
	public int predict(double[] dataPoint) {
		double decision = decisionFunction(dataPoint);
		return decision >= 0 ? 1 : -1; // Return +1 or -1 based on the decision
	}

	// Decision function
	public double decisionFunction(double[] dataPoint) {
		// select linear kernel
		double kernelResult = linearKernel(weights, dataPoint);
		return kernelResult + bias;
	}

	// Linear Kernel
	private double linearKernel(double[] x1, double[] x2) {
		double result = 0.0;
		for (int i = 0; i < x1.length; i++) {
			result += x1[i] * x2[i]; // Dot product
		}
		return result;
	}

	// Method to evaluate the accuracy of the SVM on a test set
	public double evaluate(double[][] testData) {
		int correct = 0;
		int numSamples = testData.length;
		int numFeatures = testData[0].length - 1;

		for (int i = 0; i < numSamples; i++) {
			double[] features = new double[numFeatures];
			for (int j = 0; j < numFeatures; j++) {
				features[j] = testData[i][j];
			}
			int actualLabel = (int) testData[i][numFeatures];

			int predictedLabel = predict(features);
			if (predictedLabel == actualLabel) {
				correct++;
			}
		}

		return (double) correct / numSamples;
	}
}
