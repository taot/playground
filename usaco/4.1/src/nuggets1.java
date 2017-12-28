/*
ID: libra_k1
LANG: JAVA
TASK: nuggets
*/

import java.io.*;
import java.util.BitSet;
import java.util.StringTokenizer;

class nuggets1 {

    private static String task = "nuggets";

//    static int MAX = 400;
    static int MAX = 2000000000 + 1;
    static int N;

    static int nuggets[];
    static int maxNugget = -1;

    static BitSet set = new BitSet();

    public static void main (String [] args) throws IOException {
        long start = System.currentTimeMillis();
        BufferedReader f = new BufferedReader(new FileReader(task + ".in4"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        nuggets = new int[N];
        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            nuggets[i] = Integer.parseInt(st.nextToken());
            if (nuggets[i] > maxNugget) {
                maxNugget = nuggets[i];
            }
        }

        int m = find();
        System.out.println(m);
        out.println(m);

        out.close();
        System.out.println("Duration: " + (System.currentTimeMillis() - start) + " ms");
    }

    static int find() {
        for (int n : nuggets) {
            set.set(n);
        }
        int nSeq = 0;
        int maxImpossible = 0;
        for (int i = 1; i < MAX && nSeq <= maxNugget; i++) {
            if (set.get(i)) {
                nSeq++;
                for (int n : nuggets) {
                    set.set(i + n);
                }
            } else {
                maxImpossible = i;
                nSeq = 0;
            }
        }
        return maxImpossible;
    }
}
