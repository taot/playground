/*
ID: libra_k1
LANG: JAVA
TASK: humble
*/
import java.io.*;
import java.util.*;

class humble {

    private static String task = "humble";

    static int K;
    static int N;

    static int[] primes;

    static int MAX = 2147483647 / 2;

    static int hums[] = new int[100001];
    static int nhum = 0;
    static int pindex[] = new int[100];

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
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
        find();
        System.out.println(hums[nhum-1]);
        out.println(hums[nhum-1]);

        System.out.println("duration: " + (System.currentTimeMillis() - start) + " ms");


        out.close();
    }

    static void find() {
        hums[0] = 1;
        nhum = 1;

        while (nhum < N+1) {
            for (int i = 0; i < K; i++) {
                while (hums[pindex[i]] * primes[i] <= hums[nhum-1]) {
                    pindex[i]++;
                }
            }
            int min_i = 0;
            int min = hums[pindex[min_i]] * primes[min_i];
            for (int i = 1; i < K; i++) {
                if (min > hums[pindex[i]] * primes[i]) {
                    min_i = i;
                    min = hums[pindex[i]] * primes[i];
                }
            }
            nhum++;
            hums[nhum-1] = min;
        }
    }
}
