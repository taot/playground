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
        List<Integer> hills = new ArrayList<Integer>();
        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            int h = Integer.parseInt(st.nextToken());
            hills.add(h);
        }

        Collections.sort(hills);
        int cost = 0;
        while (hills.size() > 1 && hills.get(hills.size() - 1) - hills.get(0) > 17) {
            // System.out.println(hills.size());
            // System.out.println(hills);
            int diff = hills.get(hills.size() - 1) - hills.get(0) - 17;
            if (diff > 0) {
                if (diff % 2 == 0) {
                    int x = diff / 2;
                    cost += (x * x * 2);
                } else {
                    int x = diff / 2;
                    int y = x + 1;
                    cost += (x * x + y * y);
                }
            }
            hills.remove(hills.size() - 1);
            hills.remove(0);
        }
        // System.out.println(hills.get(hills.size() - 1) - hills.get(0));
        out.println(cost);
        // System.out.println(cost);

        out.close();
    }
}
