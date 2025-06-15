import java.io.IOException;
import Utils.Dataloader;
import svm_ga.GeneticAlgorithm;
import svm_ga.TwoFoldCrossValidation;
import svm_ga.OneVsAllSVM;
import svm_ga.TestEvaluator;
import Utils.DataUtils;

public class Main {
	public static void main(String[] args) {
		// Paths to the detests
		String trainingDataPath = "data/dataSet1.csv";
		String testDataPath = "data/dataSet2.csv";

		try {
			// Load training and test data
			double[][] trainingData = Dataloader.loadCSV(trainingDataPath);
			double[][] testData = Dataloader.loadCSV(testDataPath);
            //Ensure all data are loaded
			System.out.println("Training data loaded: " + trainingData.length + " samples");
			System.out.println("Test data loaded: " + testData.length + " samples");
			// Number of Classes based on the Dataset 
			int numClasses = DataUtils.getNumberOfClasses(trainingData);
			System.out.println("Number of Classes: " + numClasses);
			
			// Normalize the data
			System.out.println("Normalizing the data...");
			trainingData = DataUtils.normalize(trainingData);
			testData = DataUtils.normalize(testData);
			System.out.println("Data normalization completed."); 

			// Initialize Genetic Algorithm for SVM hyperparameter optimization
			GeneticAlgorithm ga = new GeneticAlgorithm();

			// Optimize the SVM's C value using the GA
			System.out.println("Running Genetic Algorithm to optimize C...");
			double bestC = ga.optimizeSVM(trainingData, numClasses);
			System.out.println("Genetic Algorithm optimization completed!");
			System.out.println("Best C value found by GA: " + bestC);

			// Initialize One-vs-All SVM with the best C value
			System.out.println("Extend SVM using OneVsAllSVM and create binary classifiers for multiple classes");
			OneVsAllSVM oneVsAllSVM = new OneVsAllSVM(numClasses, bestC); 

			// Perform Two-Fold Cross-Validation on the optimized One-vs-All SVM
			System.out.println("Starting Two-Fold Cross-Validation on the optimiezed One-vs-All SVM...");
			int[][] confusionMatrix = new int[numClasses][numClasses];
			double accuracy = TwoFoldCrossValidation.performTwoFold(oneVsAllSVM, trainingData, numClasses,
					confusionMatrix);
			System.out.println("Two-Fold Cross-Validation completed!");

			//Final Accuracy on training data
			System.out.println("Final Accuracy after Two-Fold Cross-Validation: " + accuracy + "%");

			//Evaluation of the model on the test data
			TestEvaluator testEvaluator = new TestEvaluator();
			System.out.println("Evaluating on test data...");
			double testAccuracy = testEvaluator.evaluate(oneVsAllSVM, testData, numClasses);
			System.out.println("Test Accuracy: " + testAccuracy + "%");

		} catch (IOException e) {
			System.err.println("Error loading the dataset: " + e.getMessage());
		}
	}
}