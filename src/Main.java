public class Main {
    static final String dataPath = "h5a.txt";
    static final String outPath = "out.txt";
    public static void main(String[] args) {
        try {
            DataFrame df = new DataFrame(dataPath,true);
            // write your code here
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println(e.getMessage());
        }
    }
}
