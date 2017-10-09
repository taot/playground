/*
ID: libra_k1
LANG: JAVA
TASK: agrinet
*/
import java.io.*;
import java.util.*;

class agrinet {

    private static String task = "agrinet";

    private static int N;
    private static int[][] graph;

    static boolean[] intree;
    static int[] dist;
    static int cost;
    static int nInTree;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());

        graph = new int[N][N];
        intree = new boolean[N];
        dist = new int[N];
        cost = 0;
        for (int i = 0; i < N; i++) {
            intree[i] = false;
            dist[i] = Integer.MAX_VALUE;
        }

        st = new StringTokenizer(f.readLine());
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (! st.hasMoreTokens()) {
                    st = new StringTokenizer(f.readLine());
                }
                graph[i][j] = Integer.parseInt(st.nextToken());
            }
        }

        prim();

        System.out.println(cost);
        out.println(cost);
        out.close();
    }

    private static void prim() {
        intree[0] = true;
        dist[0] = 0;
        nInTree = 1;
        for (int i = 1; i < N; i++) {
            dist[i] = graph[0][i];
        }

        while (nInTree < N) {
            int min_i = -1;
            int min_dist = Integer.MAX_VALUE;
            for (int i = 0; i < N; i++) {
                if (! intree[i] && dist[i] < min_dist) {
                    min_i = i;
                    min_dist = dist[i];
                }
            }
            if (min_i < 0) {
                throw new RuntimeException("Failed");
            }
            cost += min_dist;
            intree[min_i] = true;
            nInTree++;
            for (int i = 0; i < N; i++) {
                if (i == min_i) {
                    continue;
                }
                if (dist[i] > graph[min_i][i]) {
                    dist[i] = graph[min_i][i];
                }
            }
        }
    }
}
