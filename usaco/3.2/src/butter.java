/*
ID: libra_k1
LANG: JAVA
TASK: butter
*/

import java.io.*;
import java.util.StringTokenizer;

class butter {

    private static String task = "butter";

    static int N, P, C;
    static int[][] graph;
    static int[][] dists;
    static int[] cows;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));

        // read input
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        P = Integer.parseInt(st.nextToken());
        C = Integer.parseInt(st.nextToken());
        cows = new int[N];
        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            cows[i] = Integer.parseInt(st.nextToken()) - 1;
        }
        graph = new int[P][P];
        dists = new int[P][P];
        for (int i = 0; i < P; i++) {
            for (int j = 0; j < P; j++) {
                graph[i][j] = dists[i][j] = 0;
            }
        }
        for (int i = 0; i < C; i++) {
            st = new StringTokenizer(f.readLine());
            int s = Integer.parseInt(st.nextToken()) - 1;
            int t = Integer.parseInt(st.nextToken()) - 1;
            int len = Integer.parseInt(st.nextToken());
            graph[s][t] = graph[t][s] = len;
            dists[s][t] = dists[t][s] = len;
        }

        // calculate
        floyd();

//        System.out.println("graph:");
//        printGraph(graph);
//        System.out.println("dists:");
//        printGraph(dists);

        int minSum = Integer.MAX_VALUE;
        for (int b = 0; b < P; b++) {
            int sum = 0;
            for (int i = 0; i < N; i++) {
                sum += dists[b][cows[i]];
            }
            if (sum < minSum) {
                minSum = sum;
            }
        }

        System.out.println(minSum);
        out.println(minSum);

        out.close();
    }

    static void floyd() {
        for (int k = 0; k < P; k++) {
            for (int i = 0; i < P; i++) {
                for (int j = i + 1; j < P; j++) {
//                    if (i == j) {
//                        continue;
//                    }
                    int d1 = dists[i][k];
                    int d2 = dists[k][j];
                    int d3 = dists[i][j];
                    int s = d1 + d2;
                    if (d1 > 0 && d2 > 0 && (d3 == 0 || d3 > s)) {
                        dists[i][j] = dists[j][i] = s;
                    }
                }
            }
        }
    }

    static void printGraph(int[][] graph) {
        for (int i = 0; i < P; i++) {
            for (int j = 0; j < P; j++) {
                System.out.print(graph[i][j] + " ");
            }
            System.out.println();
        }
    }

}
