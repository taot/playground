/*
ID: libra_k1
LANG: JAVA
TASK: skidesign
*/
import java.io.*;
import java.util.*;

class skidesign {

    private static String task = "skidesign";

    private static int N;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        int[] hills = new int[N];
        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            hills[i] = Integer.parseInt(st.nextToken());
        }

        int minCost = Integer.MAX_VALUE;
        for (int min = 0; min <= 100; min++) {
            int cost = 0;
            for (int i = 0; i < N; i++) {
                cost += calcCost(hills[i], min);
            }
            if (cost < minCost) {
                minCost = cost;
            }
        }

        // System.out.println(hills.get(hills.size() - 1) - hills.get(0));
        out.println(minCost);
        // System.out.println(cost);

        out.close();
    }

    private static int calcCost(int height, int min) {
        if (height < min) {
            int x = min - height;
            return x * x;
        }
        if (height > min + 17) {
            int x = height - min - 17;
            return x * x;
        }
        return 0;
    }
}
