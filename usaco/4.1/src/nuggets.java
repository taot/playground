/*
ID: libra_k1
LANG: JAVA
TASK: nuggets
*/
import java.io.*;
import java.util.*;

class nuggets {

    private static String task = "nuggets";

//    static int MAX = 400;
    static int MAX = 2000000000 + 1;
    static int N;

    static int nuggets[];
    static int minNugget = Integer.MAX_VALUE;

    public static void main (String [] args) throws IOException {
        long start = System.currentTimeMillis();
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        nuggets = new int[N];
        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            nuggets[i] = Integer.parseInt(st.nextToken());
            if (nuggets[i] < minNugget) {
                minNugget = nuggets[i];
            }
        }

        if (gcd(nuggets) != 1) {
            System.out.println(0);
            out.println(0);
        } else {
            int m = bfs();
            System.out.println(m);
            out.println(m);
        }

//        int g = gcd(new int[] { 3, 6, 9 });
//        System.out.println(g);

        out.close();
        System.out.println("Duration: " + (System.currentTimeMillis() - start) + " ms");
    }

    static int bfs() {
        PriorityQueue<Integer> q = new PriorityQueue<>();
        q.add(0);
        BitSet visited = new BitSet();
        visited.set(0);
        Integer m;
        while ((m = q.poll()) != null && m < MAX) {
            boolean vac = m == 0 ? true : ! visited.get(m - 1);
            for (int i = 2; i < minNugget && m - i >= 0; i++) {
                if (! visited.get(m - i)) {
                    vac = true;
                    break;
                }
            }
            if (! vac) {
                break;
            }
            for (int n : nuggets) {
                int n1 = n + m;
                if (visited.get(n1)) {
                    continue;
                }
                visited.set(n1);
                q.add(n1);
            }
        }
        return m - minNugget;
    }

    static int gcd(int[] arr) {
        int g = arr[0];
        for (int i = 1; i < arr.length; i++) {
            g = gcd(g, arr[i]);
        }
        return g;
    }

    static int gcd(int a, int b) {
        while (b > 0) {
            int c = a % b;
            a = b;
            b = c;
        }
        return a;
    }
}
