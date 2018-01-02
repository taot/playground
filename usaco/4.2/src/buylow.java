/*
ID: libra_k1
LANG: JAVA
TASK: buylow
*/
import java.io.*;
import java.math.BigInteger;
import java.util.*;

class buylow {

    private static String task = "buylow";

    static int N;
    static int[] series;
    static int[] lengths;
    static BigInteger[] counts;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        series = new int[N+1];
        lengths = new int[N+1];
        counts = new BigInteger[N+1];
        st = new StringTokenizer(f.readLine());
        for (int i = 0; i < N; i++) {
            if (! st.hasMoreElements()) {
                st = new StringTokenizer(f.readLine());
            }
            series[i] = Integer.parseInt(st.nextToken());

        }
        series[N] = 0;

        dp();

//        printArr(series);
//        printArr(lengths);

        int len = lengths[N] - 1;
        BigInteger cnt = counts[N];
        System.out.println(len + " " + cnt);
        out.println(len + " " + cnt);

        out.close();
    }

    static void dp() {
        Set<Integer> visited = new HashSet<>();
        for (int i = 0; i < N + 1; i++) {
            int m = 0;
            BigInteger c = new BigInteger("1");
            visited.clear();
            for (int j = i - 1; j >= 0; j--) {
                if (series[i] >= series[j]) {
                    continue;
                }
                if (lengths[j] > m) {
                    m = lengths[j];
                    visited.clear();
                    c = counts[j];
                    visited.add(series[j]);

                } else if (lengths[j] == m) {
                    if (! visited.contains(series[j])) {
                        c = c.add(counts[j]);
                        visited.add(series[j]);
                    }
                }
            }
            lengths[i] = m + 1;
            counts[i] = c;
        }
    }

    static void printArr(int[] a) {
        for (int i = 0; i < a.length; i++) {
            System.out.print(padding(a[i]) + " ");
        }
        System.out.println();
    }

    static String padding(int x) {
        String s = String.valueOf(x);
        int l = s.length();
        for (int i = 0; i < 3 - l; i++) {
            s = " " + s;
        }
        return s;
    }

}
