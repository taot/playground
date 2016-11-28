/*
ID: libra_k1
LANG: JAVA
TASK: barn1
*/
import java.io.*;
import java.util.*;

class barn1 {

    private static String task = "barn1";

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        int maxNumOfBoards = Integer.parseInt(st.nextToken());
        int numOfStalls = Integer.parseInt(st.nextToken());
        int numOfCows = Integer.parseInt(st.nextToken());

        int[] cows = new int[numOfCows];
        for (int i = 0; i < numOfCows; i++) {
            st = new StringTokenizer(f.readLine());
            cows[i] = Integer.parseInt(st.nextToken());
        }
        Arrays.sort(cows);

        List<Integer> holes = new ArrayList<Integer>();
        for (int i = 0; i < numOfCows - 1; i++) {
            int d = cows[i + 1] - cows[i];
            if (d > 1) {
                holes.add(d - 1);
            }
        }
        Collections.sort(holes);
        // System.out.println(holes);
        int sum = 0;
        for (int i = 0; i < (holes.size() - maxNumOfBoards + 1); i++) {
            sum += holes.get(i);
        }
        sum += numOfCows;

        out.println(sum);
        out.close();
    }
}
