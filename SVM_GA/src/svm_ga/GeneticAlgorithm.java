package svm_ga;

import java.util.Random;
import Utils.sortIndicesByFitness;

public class GeneticAlgorithm {
	private Random rand = new Random();
	private final int POPULATION_SIZE = 10; // Number of individuals in the population
	private final double MUTATION_RATE = 0.1; // Mutation rate
	private final int MAX_GENERATIONS = 10; // Number of generations

	// Optimize SVM using GA
	public double optimizeSVM(double[][] trainingData, int numClasses) {
		// Initialize population 
		double[] population = initializePopulation();

		// Run evolution loop
		for (int generation = 0; generation < MAX_GENERATIONS; generation++) {
			System.out.println("Generation " + generation + " - evaluating population...");

			int[][] confusionMatrix = new int[numClasses][numClasses];
			// Evaluate fitness of each individual in the population
			double[] fitness = evaluatePopulation(population, trainingData, numClasses, confusionMatrix);

			// Print confusion matrix for the current generation
			System.out.println("Confusion Matrix for Generation " + generation + ":");
			printConfusionMatrix(confusionMatrix, numClasses);

			// Evolve population (selection, crossover, mutation)
			population = evolvePopulation(population, fitness);
		}

		// Select the best C value after optimization
		return selectBestHyperparameter(population, trainingData, numClasses);
	}

	// Initialize population with random C values in the range [0.01, 1000]
	private double[] initializePopulation() {
		double[] population = new double[POPULATION_SIZE];
		for (int i = 0; i < POPULATION_SIZE; i++) {
			population[i] = Math.pow(10, -2 + (rand.nextDouble() * 5));
		}
		return population;
	}

	// Evaluate fitness of each SVM in the population
	private double[] evaluatePopulation(double[] population, double[][] trainingData, int numClasses,
			int[][] confusionMatrix) {
		double[] fitness = new double[population.length];

		for (int i = 0; i < population.length; i++) {
			System.out.println("Evaluating C = " + population[i]);
			// Initialize One-vs-All SVM with current C value
			OneVsAllSVM oneVsAllSVM = new OneVsAllSVM(numClasses, population[i]);
			// Perform two-fold cross-validation
			fitness[i] = TwoFoldCrossValidation.performTwoFold(oneVsAllSVM, trainingData, numClasses, confusionMatrix);
			System.out.println("Fitness for C = " + population[i] + ": " + fitness[i] + "%");
		}
		return fitness;
	}

	private void printConfusionMatrix(int[][] confusionMatrix, int numClasses) {
		System.out.println("Confusion Matrix");

		// Print column headers
		System.out.print("     ");
		for (int i = 0; i < numClasses; i++) {
			System.out.printf("%6d", i);
		}
		System.out.println();

		// Print top border
		System.out.print("     ");
		for (int i = 0; i < numClasses; i++) {
			System.out.print("------");
		}
		System.out.println();

		// Print matrix rows with row labels
		for (int i = 0; i < numClasses; i++) {
			System.out.printf("%4d |", i); 
			for (int j = 0; j < numClasses; j++) {
				System.out.printf("%6d", confusionMatrix[i][j]);
			}
			System.out.println();
		}

		// Print bottom border
		System.out.print("     ");
		for (int i = 0; i < numClasses; i++) {
			System.out.print("------");
		}
		System.out.println();
	}

	// Evolve population: selection, crossover, and mutation
	private double[] evolvePopulation(double[] population, double[] fitness) {
		double[] newPopulation = new double[population.length];

		// Preserve the top 2 individuals (elitism)
		int elitismCount = 2;

		// Sort the indices based on fitness
		sortIndicesByFitness sorter = new sortIndicesByFitness();
		int[] sortedIndices = sorter.sort(fitness); 

		// Preserve the top 2 elite individuals
		for (int i = 0; i < elitismCount; i++) {
			newPopulation[i] = population[sortedIndices[i]];
		}

		// Fill the rest of the new population through selection, crossover, and mutation
		for (int i = elitismCount; i < population.length; i++) {
			// Selection: select two parents using roulette wheel selection
			double parent1 = selectParent(population, fitness);
			double parent2 = selectParent(population, fitness);

			// Crossover: create a child by averaging the parents
			double child = (parent1 + parent2) / 2.0;

			// Mutation: randomly mutate the child
			if (rand.nextDouble() < MUTATION_RATE) {
				child += rand.nextGaussian(); // Small mutation
			}
			// Ensure C stays in the [0.01, 1000] range
			child = Math.max(0.01, Math.min(1000, child));

			// Add child to the new population
			newPopulation[i] = child;
		}

		return newPopulation;
	}

	// Select a parent using roulette wheel selection based on fitness
	private double selectParent(double[] population, double[] fitness) {
		double totalFitness = 0;
		for (double f : fitness) {
			totalFitness += f;
		}

		double randomValue = rand.nextDouble() * totalFitness;
		double cumulativeFitness = 0;

		for (int i = 0; i < population.length; i++) {
			cumulativeFitness += fitness[i];
			if (cumulativeFitness >= randomValue) {
				return population[i];
			}
		}

		return population[population.length - 1];
	}

	private double selectBestHyperparameter(double[] population, double[][] trainingData, int numClasses) {
		double bestFitness = Double.NEGATIVE_INFINITY;
		double bestC = population[0];

		int[][] confusionMatrix = new int[numClasses][numClasses];

		// Find the best fitness and corresponding C value
		for (int i = 0; i < population.length; i++) {
			double fitness = evaluateFitness(population[i], trainingData, numClasses, confusionMatrix);
			if (fitness > bestFitness) {
				bestFitness = fitness;
				bestC = population[i];
			}
		}

		// Print final confusion matrix for the best C value
		System.out.println("Best C found with fitness: " + bestFitness + "%");
		printConfusionMatrix(confusionMatrix, numClasses);

		return bestC;
	}

	//evaluate fitness for a given C value
	private double evaluateFitness(double C, double[][] trainingData, int numClasses, int[][] confusionMatrix) {
		// Initialize One-vs-All SVM with the given number of classes and C value
		OneVsAllSVM oneVsAllSVM = new OneVsAllSVM(numClasses, C);

		// Perform two-fold cross-validation and pass the confusion matrix for tracking
		return TwoFoldCrossValidation.performTwoFold(oneVsAllSVM, trainingData, numClasses, confusionMatrix);
	}

}
