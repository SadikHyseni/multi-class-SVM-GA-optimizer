package Utils;

public class DataUtils {

	// Method to standardize the dataset (center and scale to have mean 0 and std 1)
	public static double[][] normalize(double[][] data) {
		int numSamples = data.length; // 64 samples
		int numFeatures = data[0].length - 1; // 2809 features (excluding the label column)

		// Step 1: Compute the Euclidean center (mean) for each feature
		double[] center = new double[numFeatures];
		for (int i = 0; i < numSamples; i++) {
			for (int j = 0; j < numFeatures; j++) {
				center[j] += data[i][j];
			}
		}
		for (int j = 0; j < numFeatures; j++) {
			center[j] /= numSamples; // Compute the mean for each feature
		}

		// Step 2: Compute the standard deviation for each feature
		double[] stdDev = new double[numFeatures];
		for (int i = 0; i < numSamples; i++) {
			for (int j = 0; j < numFeatures; j++) {
				stdDev[j] += Math.pow(data[i][j] - center[j], 2);
			}
		}
		for (int j = 0; j < numFeatures; j++) {
			stdDev[j] = Math.sqrt(stdDev[j] / numSamples); // Compute the standard deviation for each feature
		}

		// Step 3: Standardize the data (center and scale to have mean 0 and std 1)
		double[][] standardizedData = new double[numSamples][numFeatures + 1]; // +1 for the label column
		for (int i = 0; i < numSamples; i++) {
			for (int j = 0; j < numFeatures; j++) {
				// Center the data and scale it to have mean 0 and std 1
				if (stdDev[j] != 0) {
					standardizedData[i][j] = (data[i][j] - center[j]) / stdDev[j];
				} else {
					standardizedData[i][j] = data[i][j] - center[j]; // Handle case where stdDev[j] is 0 (constant
																		// feature)
				}
			}
			standardizedData[i][numFeatures] = data[i][numFeatures]; // Copy the label
		}

		return standardizedData;
	}
	
	  // Get the number of unique classes based on the labels in the dataset
    public static int getNumberOfClasses(double[][] data) {
        int numSamples = data.length;
        int numFeatures = data[0].length - 1; 
        int maxLabel = Integer.MIN_VALUE;  // store highest label value
        
        //maximum label value
        for (int i = 0; i < numSamples; i++) {
            int label = (int) data[i][numFeatures];
            if (label > maxLabel) {
                maxLabel = label;
            }
        }
        
        boolean[] labelExists = new boolean[maxLabel + 1];
        
        // Loop through the dataset
        for (int i = 0; i < numSamples; i++) {
            int label = (int) data[i][numFeatures];
            labelExists[label] = true;
        }

        // Count the number of true values
        int numClasses = 0;
        for (boolean exists : labelExists) {
            if (exists) {
                numClasses++; 
            }
        }

        return numClasses;
    }
	
	
	
}