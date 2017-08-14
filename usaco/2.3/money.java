/*
ID: libra_k1
LANG: JAVA
TASK: money
*/
import java.io.*;
import java.util.*;

class money {

    private static String task = "money";

    static int V;
    static int N;
    static int[] C;
    static long[][] P;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        V = Integer.parseInt(st.nextToken());
        N = Integer.parseInt(st.nextToken());
        Set<Integer> set = new TreeSet<>();


        {
            st = new StringTokenizer(f.readLine());
            int i = 0;
            while (i < V) {
                while (st.hasMoreTokens() && i < V) {
                    int x = Integer.parseInt(st.nextToken());
                    set.add(x);
                }
                String l = f.readLine();
                if (l == null) {
                    break;
                }
                st = new StringTokenizer(l);
            }
        }

        // st = new StringTokenizer(f.readLine());
        //
        // while (st.hasMoreTokens() && )
        // for (int i = 0; i < V; i++) {
        //     C[i] = Integer.parseInt(st.nextToken());
        // }
        V = set.size();
        C = new int[V];
        {
            int i = 0;
            for (Integer x : set) {
                C[i] = x;
                i++;
            }
        }
        P = new long[N+1][V+1];

        for (int i = 0; i <= V; i++) {
            P[0][i] = 1;
        }
        for (int n = 1; n <= N; n++) {
            P[n][0] = 0;
        }


        for (int n = 1; n <= N; n++) {
            for (int i = 1; i <= V; i++) {
                long sum = 0;
                int j = 0;
                while (j * C[i-1] <= n) {
                    sum += P[n - C[i-1] * j][i-1];
                    j++;
                    // System.out.println("i = " + i + " j = " + j);
                    // System.out.println("sum = " + sum);
                }
                P[n][i] = sum;
            }

        }
        // System.out.println("hello");

        out.println(P[N][V]);
        // printP();

        out.close();
    }

    static void printP() {
        for (int n = 0; n <= N; n++) {
            for (int i = 0; i <= V; i++) {
                System.out.print(P[n][i] + " ");
            }
            System.out.println();
        }
    }
}
