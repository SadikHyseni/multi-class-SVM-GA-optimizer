package Utils;

public class sortIndicesByFitness {
    public int[] sort(double[] fitness) {
        int n = fitness.length;
        int[] indices = new int[n];

        // Initialize the indices array with the original indices
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }

        // Bubble sort to sort the indices based on fitness values in descending order
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (fitness[indices[j]] < fitness[indices[j + 1]]) {
                    // Swap indices[j] and indices[j + 1]
                    int temp = indices[j];
                    indices[j] = indices[j + 1];
                    indices[j + 1] = temp;
                }
            }
        }

        return indices;
    }
}