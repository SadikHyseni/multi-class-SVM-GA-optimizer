package svm_ga;

public class TwoFoldCrossValidation {

	public static double performTwoFold(OneVsAllSVM oneVsAllSVM, double[][] data, int numClasses,
			int[][] confusionMatrix) {
		int dataSize = data.length;
		int splitIndex = dataSize / 2;

		// First fold
		double[][] firstTrain = new double[splitIndex][];
		double[][] firstTest = new double[dataSize - splitIndex][];
		System.arraycopy(data, 0, firstTrain, 0, splitIndex);
		System.arraycopy(data, splitIndex, firstTest, 0, dataSize - splitIndex);

		// Train and evaluate for the first fold
		System.out.println("Performing first fold...");
		oneVsAllSVM.train(firstTrain);
		int[] predictionsFirstFold = new int[firstTest.length];
		for (int i = 0; i < firstTest.length; i++) {
			predictionsFirstFold[i] = oneVsAllSVM.predict(firstTest[i]);
		}
		double accuracyFirstFold = updateConfusionMatrixAndGetAccuracy(predictionsFirstFold, firstTest, confusionMatrix,
				numClasses);

		// Second fold
		double[][] secondTrain = new double[dataSize - splitIndex][];
		double[][] secondTest = new double[splitIndex][];
		System.arraycopy(data, splitIndex, secondTrain, 0, dataSize - splitIndex);
		System.arraycopy(data, 0, secondTest, 0, splitIndex);

		// Train and evaluate for the second fold
		System.out.println("Performing second fold...");
		oneVsAllSVM.train(secondTrain); 
		int[] predictionsSecondFold = new int[secondTest.length];
		for (int i = 0; i < secondTest.length; i++) {
			predictionsSecondFold[i] = oneVsAllSVM.predict(secondTest[i]);
		}
		double accuracySecondFold = updateConfusionMatrixAndGetAccuracy(predictionsSecondFold, secondTest,
				confusionMatrix, numClasses);

		// Compute the average accuracy
		double averageAccuracy = (accuracyFirstFold + accuracySecondFold) / 2.0;

		return averageAccuracy;
	}

	// update the confusion matrix and calculate accuracy
	private static double updateConfusionMatrixAndGetAccuracy(int[] predictions, double[][] actualTestData,
			int[][] confusionMatrix, int numClasses) {
		int correctPredictions = 0;
		int totalSamples = predictions.length;

		for (int i = 0; i < totalSamples; i++) {
			int actualLabel = (int) actualTestData[i][actualTestData[i].length - 1]; 
			int predictedLabel = predictions[i];

			// Update the confusion matrix
			confusionMatrix[actualLabel][predictedLabel]++;

			if (actualLabel == predictedLabel) {
				correctPredictions++;
			}
		}

		// Return accuracy as the percentage of correct predictions
		return ((double) correctPredictions / totalSamples) * 100.0;
	}

}