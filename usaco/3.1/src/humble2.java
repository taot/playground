/*
ID: libra_k1
LANG: JAVA
TASK: humble
*/

import java.io.*;
import java.util.StringTokenizer;
import java.util.TreeSet;

class humble2 {

    private static String task = "humble";

    static int K;
    static int N;

    static long[] primes;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        K = Integer.parseInt(st.nextToken());
        N = Integer.parseInt(st.nextToken());

        st = new StringTokenizer(f.readLine());
        primes = new long[K];
        for (int i = 0; i < K; i++) {
            primes[i] = Integer.parseInt(st.nextToken());
        }

        long start = System.currentTimeMillis();
        long res = bfs();
        System.out.println("res = " + res);
        System.out.println("duration: " + (System.currentTimeMillis() - start) + " ms");
        out.println(res);

        out.close();
    }

    static long bfs() {
        int count = -1;
        TreeSet<Long> set = new TreeSet<>();
//        PriorityQueue<Long> set = new PriorityQueue<>();
        set.add(1L);

        while (true) {
            long n = set.pollFirst();
            count++;
//            System.out.println(n);
            if (count == N) {
                return n;
            }
//            List<Long> list = new ArrayList<>();
            for (int i = 0; i < K; i++) {
                long k = primes[i] * n;
                if (k > 0 && ! set.contains(k)) {
                    set.add(k);
                }
            }
//            set.addAll(list);
        }
    }
}
