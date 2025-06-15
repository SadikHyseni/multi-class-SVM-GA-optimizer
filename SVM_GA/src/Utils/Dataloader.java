package Utils;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
public class Dataloader {

	//load CSV file and convert it to a 2D array
	public static double[][] loadCSV(String filePath) throws IOException {
		InputStream inputStream = Utils.Dataloader.class.getClassLoader().getResourceAsStream(filePath);
		// Check if file was found
		if (inputStream == null) {
			throw new IOException("File not found: " + filePath);
		}
		BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
		String line;

		int rowCount = 0;
		int columnCount = 0;
		while ((line = br.readLine()) != null) {
			if (rowCount == 0) {
				columnCount = line.split(",").length; //number of columns from first row
			}
			rowCount++;
		}
		br.close();

		//read the data into the array
		inputStream = Utils.Dataloader.class.getClassLoader().getResourceAsStream(filePath);
		br = new BufferedReader(new InputStreamReader(inputStream));
		double[][] data = new double[rowCount][columnCount];
		int row = 0;
		while ((line = br.readLine()) != null) {
			String[] tokens = line.split(",");
			for (int col = 0; col < tokens.length; col++) {
				data[row][col] = Double.parseDouble(tokens[col]);
			}
			row++;
		}
		br.close();
		return data;
	}
}
