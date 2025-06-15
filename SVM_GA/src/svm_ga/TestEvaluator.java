package svm_ga;

public class TestEvaluator {

    // Method to evaluate the model on the test data
    public double evaluate(OneVsAllSVM oneVsAllSVM, double[][] testData, int numClasses) {
        int correct = 0;
        int numSamples = testData.length;
        int numFeatures = testData[0].length - 1; // Exclude label column

        // Confusion matrix for test data
        int[][] confusionMatrix = new int[numClasses][numClasses];

        // Variables for metrics
        double[] classPrecision = new double[numClasses];
        double[] classRecall = new double[numClasses];
        double[] classF1Score = new double[numClasses];
        double[] classTruePositive = new double[numClasses];
        double[] classFalsePositive = new double[numClasses];
        double[] classFalseNegative = new double[numClasses];

        // Loop through all test samples
        for (int i = 0; i < numSamples; i++) {
            double[] features = new double[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                features[j] = testData[i][j];
            }

            int actualLabel = (int) testData[i][numFeatures]; // Get actual class label
            int predictedLabel = oneVsAllSVM.predict(features); // Predict the class label

            // Update confusion matrix
            confusionMatrix[actualLabel][predictedLabel]++;

            // Calculate accuracy
            if (predictedLabel == actualLabel) {
                correct++;
            }

            // Calculate additional metrics (precision, recall, F1-score)
            if (predictedLabel == actualLabel) {
                classTruePositive[actualLabel]++;
            } else {
                classFalsePositive[predictedLabel]++;
                classFalseNegative[actualLabel]++;
            }
        }

        // Print confusion matrix
        printConfusionMatrix(confusionMatrix, numClasses);

        // Calculate precision, recall, F1 score for each class
        for (int i = 0; i < numClasses; i++) {
            double precision = (classTruePositive[i] + classFalsePositive[i] == 0) ? 0 : (classTruePositive[i] / (classTruePositive[i] + classFalsePositive[i]));
            double recall = (classTruePositive[i] + classFalseNegative[i] == 0) ? 0 : (classTruePositive[i] / (classTruePositive[i] + classFalseNegative[i]));
            double f1Score = (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall);

            classPrecision[i] = precision;
            classRecall[i] = recall;
            classF1Score[i] = f1Score;
        }

        // Print metrics for each class
        printMetrics(classPrecision, classRecall, classF1Score, numClasses);

        // Calculate and return the test accuracy
        return (double) correct / numSamples * 100.0; // Return percentage
    }

    // Helper method to print the confusion matrix in a formatted way
    private void printConfusionMatrix(int[][] confusionMatrix, int numClasses) {
        System.out.println("Confusion Matrix for Test Data:");
        System.out.println("------------------------------------------------");
        System.out.print("   ");
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("%5d", i);
        }
        System.out.println();
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("%2d |", i);
            for (int j = 0; j < numClasses; j++) {
                System.out.printf("%5d", confusionMatrix[i][j]);
            }
            System.out.println();
        }
        System.out.println("------------------------------------------------");
    }

    // Helper method to print precision, recall, and F1 score for each class
    private void printMetrics(double[] precision, double[] recall, double[] f1Score, int numClasses) {
        System.out.println("Metrics for Each Class:");
        System.out.println("------------------------------------------------");
        System.out.printf("%-5s %-12s %-12s %-12s\n", "Class", "Precision", "Recall", "F1-Score");
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("%-5d %-12.4f %-12.4f %-12.4f\n", i, precision[i], recall[i], f1Score[i]);
        }
        System.out.println("------------------------------------------------");
    }
}
