package svm_ga;

public class OneVsAllSVM {
	private SVM[] classifiers; 
	private int numClasses;
	private double C;

	public OneVsAllSVM(int numClasses, double C) {
		this.numClasses = numClasses;
		this.C = C;
		this.classifiers = new SVM[numClasses];
	}

	public void train(double[][] trainingData) {
		// Standardize the training data before training
		trainingData = standardize(trainingData);

		System.out.println("Training SVM ");
		for (int classLabel = 0; classLabel < numClasses; classLabel++) {
			double[][] modifiedTrainingData = modifyLabels(trainingData, classLabel);
			classifiers[classLabel] = new SVM(C);
			classifiers[classLabel].train(modifiedTrainingData);
		}
	}

	// Modify the labels for the One-vs-All classification problem
	// +1 for the current class, -1 for other classes
	private double[][] modifyLabels(double[][] data, int classLabel) {
		double[][] modifiedData = new double[data.length][];
		for (int i = 0; i < data.length; i++) {
			modifiedData[i] = data[i].clone();
			int label = (int) data[i][data[i].length - 1];
			modifiedData[i][data[i].length - 1] = (label == classLabel) ? 1 : -1;
		}
		return modifiedData;
	}

	// Standardize the data (center and scale to have mean 0 and std 1)
	private double[][] standardize(double[][] data) {
		int numSamples = data.length;
		int numFeatures = data[0].length - 1; 

		// Compute the Euclidean center for each feature
		double[] center = new double[numFeatures];
		for (int i = 0; i < numSamples; i++) {
			for (int j = 0; j < numFeatures; j++) {
				center[j] += data[i][j];
			}
		}
		for (int j = 0; j < numFeatures; j++) {
			center[j] /= numSamples;
		}

		// Compute the standard deviation for each feature
		double[] stdDev = new double[numFeatures];
		for (int i = 0; i < numSamples; i++) {
			for (int j = 0; j < numFeatures; j++) {
				stdDev[j] += Math.pow(data[i][j] - center[j], 2);
			}
		}
		for (int j = 0; j < numFeatures; j++) {
			stdDev[j] = Math.sqrt(stdDev[j] / numSamples);
		}

		// Standardize the data to have mean 0 and std 1
		double[][] standardizedData = new double[numSamples][numFeatures + 1]; // +1 for the label column
		for (int i = 0; i < numSamples; i++) {
			for (int j = 0; j < numFeatures; j++) {
				if (stdDev[j] != 0) {
					standardizedData[i][j] = (data[i][j] - center[j]) / stdDev[j];
				} else {
					standardizedData[i][j] = data[i][j] - center[j]; //if stdDev[j] is 0 
				}
			}
			standardizedData[i][numFeatures] = data[i][numFeatures];
		}

		return standardizedData;
	}

	// Predict the class for a single data point using One-vs-All
	public int predict(double[] dataPoint) {
		double[] decisionValues = new double[numClasses];

		// Get decision values from each SVM classifier
		for (int classLabel = 0; classLabel < numClasses; classLabel++) {
			decisionValues[classLabel] = classifiers[classLabel].decisionFunction(dataPoint);
		}

		// Get highest decision value
		int bestClass = 0;
		for (int i = 1; i < numClasses; i++) {
			if (decisionValues[i] > decisionValues[bestClass]) {
				bestClass = i;
			}
		}

		return bestClass;
	}

}