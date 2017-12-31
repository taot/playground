/*
ID: libra_k1
LANG: JAVA
TASK: ditch
*/

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

class ditch1 {

    private static String task = "ditch";

    static int N, M;

    static final int MAX_N = 200;       // # of edges
    static final int MAX_M = 200;       // # of vertices

    static int[][] graph = new int[MAX_M][MAX_M];
    static int[] prev = new int[MAX_M];
    static int[] flow = new int[MAX_M];
    static boolean[] visited = new boolean[MAX_M];


    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in10"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        M = Integer.parseInt(st.nextToken());

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < M; j++) {
                graph[i][j] = 0;
            }
        }
        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            int s = Integer.parseInt(st.nextToken()) - 1;
            int e = Integer.parseInt(st.nextToken()) - 1;
            int c = Integer.parseInt(st.nextToken());
            graph[s][e] = c;
        }

        int sum = maximumFlow();
        System.out.println(sum);
        out.println(sum);

        out.close();
    }

    static int maximumFlow() {
        int sum = 0;
        while (true) {
            int maxFlow = findCapPath();
            if (maxFlow == 0) {
                break;
            }
            sum += maxFlow;
            int cur = M - 1;
            List<Integer> list = new ArrayList<>();
            while (cur != 0) {
                list.add(cur);
                int pre = prev[cur];
                graph[pre][cur] -= maxFlow;
                graph[cur][pre] += maxFlow;
                cur = pre;
            }
            System.out.print(maxFlow + ": 0");
            for (int i = list.size() - 1; i >= 0; i--) {
                System.out.print(" -> " + list.get(i));
            }
            System.out.println();
        }
        return sum;
    }

    static int findCapPath() {
        for (int i = 0; i < M; i++) {
            flow[i] = 0;
            visited[i] = false;
        }
        flow[0] = Integer.MAX_VALUE;
        prev[0] = -1;
        while (true) {
            int maxCap = -1;
            int maxVertex = -1;

            for (int i = 0; i < M; i++) {
                if (visited[i]) {
                    continue;
                }
                if (flow[i] > maxCap) {
                    maxCap = flow[i];
                    maxVertex = i;
                }
            }

            if (maxVertex < 0) {
                break;
            }
            visited[maxVertex] = true;
//            if (maxVertex == M - 1) {
//                break;
//            }

            for (int i = 0; i < M; i++) {
                if (graph[maxVertex][i] == 0 || i == maxVertex) {
                    continue;
                }
                if (flow[i] < Math.min(graph[maxVertex][i], maxCap)) {
                    flow[i] = Math.min(graph[maxVertex][i], maxCap);
                    prev[i] = maxVertex;
                }
            }
        }

        return flow[M-1];
    }

}
