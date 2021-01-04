import java.io.*;

// Reads the input data
public class DataFrame {
    int nInputNodes, nHiddenNodes, nOutputNodes, nRows;
    double[][] data;

    DataFrame(String dataPath, boolean normalize) throws Exception {
        BufferedReader br;
        try {
            br = new BufferedReader(new FileReader(dataPath));
        } catch (FileNotFoundException e) {
            throw new Exception(dataPath + " is not exist");
        }

        try {
            String[] firstLine = br.readLine().split(" ");
            nInputNodes = Integer.parseInt(firstLine[0].trim());
            nHiddenNodes = Integer.parseInt(firstLine[1].trim());
            nOutputNodes = Integer.parseInt(firstLine[2].trim());
        } catch (Exception e) {
            throw new Exception("Cannot process the first line");
        }
        try {
            nRows = Integer.parseInt(br.readLine().trim());
        } catch (Exception e) {
            throw new Exception("Cannot process the second line");
        }
        try {
            data = readDataFromFile(br);
        } catch (Exception e) {
            throw new Exception("Cannot read the data");
        }

        if (normalize)
            data = normalizeData();

    }

    double[][] readDataFromFile(BufferedReader br) throws IOException {
        double[][] data = new double[nRows][nInputNodes + nOutputNodes];
        int i = 0;
        for (String line; (line = br.readLine()) != null; ++i) {
            String[] lineData = line.split(" ");
            assert (lineData.length == nInputNodes + nOutputNodes);
            int currCol = 0;
            for (String lineDatum : lineData) {
                String curr = lineDatum.trim();
                if (!curr.isEmpty())
                    data[i][currCol++] = Double.parseDouble(curr);
            }
        }
        return data;
    }

    private double[][] normalizeData() {
        double[][] ret = new double[data.length][data[0].length];
        double[] colMean = new double[data[0].length];
        double[] colStd = new double[data[0].length];

        // get mean
        for (double[] datum : data) {
            for (int j = 0; j < data[0].length; ++j) {
                colMean[j] += datum[j];
            }
        }
        for (int i = 0; i < colMean.length; ++i)
            colMean[i] /= nRows;

        // get standard deviation
        for (double[] datum : data) {
            for (int j = 0; j < data[0].length; ++j) {
                double sub = datum[j] - colMean[j];
                colStd[j] += sub * sub;
            }
        }
        for (int i = 0; i < colMean.length; ++i)
            colStd[i] = Math.sqrt(colStd[i] / nRows);

        // normalize
        for (int i = 0; i < data.length; ++i) {
            for (int j = 0; j < data[0].length; ++j) {
                ret[i][j] = (data[i][j] - colMean[j]) / colStd[j];
            }
        }

        return ret;
    }
}
