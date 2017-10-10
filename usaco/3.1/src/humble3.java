/*
ID: libra_k1
LANG: JAVA
TASK: humble
*/

import java.io.*;
import java.util.StringTokenizer;

class humble3 {

    private static String task = "humble";

    static int K;
    static int N;

    static int[] primes;

    static int[] prods;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in2"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        K = Integer.parseInt(st.nextToken());
        N = Integer.parseInt(st.nextToken());

        st = new StringTokenizer(f.readLine());
        primes = new int[K];
        for (int i = 0; i < K; i++) {
            primes[i] = Integer.parseInt(st.nextToken());
        }

        long start = System.currentTimeMillis();
        int res = dp();
        System.out.println("duration: " + (System.currentTimeMillis() - start) + " ms");

        System.out.println(res);

        out.close();
    }

    static int MAX = 2147483647 / 4;

    static int dp() {
        prods = new int[MAX];
        prods[0] = 0;
        prods[1] = 1;
        int count = 0;
        for (int i = 1; i < MAX-1; i++) {
            for (int j = 0; j < K; j++) {
                int p = primes[j];
                if (i % p == 0 && prods[i / p] == 1) {
                    prods[i] = 1;
                    count++;
                    if (count == N) {
                        return i;
                    } else {
                        break;
                    }
                }
            }
        }
        return -1;
    }
}
