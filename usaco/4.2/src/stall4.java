/*
ID: libra_k1
LANG: JAVA
TASK: stall4
*/
import java.io.*;
import java.util.*;

class stall4 {

    private static String task = "stall4";

    static int N, M;

    // source is 0, sink is N + M + 1
    static List<Edge>[] graph;

    static Edge[] prev;
    static int[] flow;
    static boolean[] visited;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        M = Integer.parseInt(st.nextToken());
        graph = new List[N + M + 2];
        prev = new Edge[N + M + 2];
        flow = new int[N + M + 2];
        visited = new boolean[N + M + 2];
        for (int i = 0; i < graph.length; i++) {
            graph[i] = new ArrayList<>();
        }

        for (int i = 0; i < N; i++) {
            int cow = i + 1;
            graph[0].add(new Edge(0, cow, 1));
            st = new StringTokenizer(f.readLine());
            int n = Integer.parseInt(st.nextToken());
            for (int j = 0; j < n; j++) {
                int s = Integer.parseInt(st.nextToken());
                graph[cow].add(new Edge(cow, s + N, 1));
            }
        }
        for (int i = 0; i < M; i++) {
            graph[i + N + 1].add(new Edge(i + N + 1, N + M + 1, 1));
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
            int cur = N + M + 1;
            while (cur != 0) {
                Edge pre = prev[cur];
                pre.cap -= maxFlow;
                Edge r = getReverseEdge(pre);
                r.cap += maxFlow;
                cur = pre.src;
            }
        }
        return sum;
    }

    static Edge getReverseEdge(Edge e) {
        for (Edge i : graph[e.dst]) {
            if (i.dst == e.src) {
                return i;
            }
        }
        Edge r = new Edge(e.dst, e.src, 0);
        graph[e.dst].add(r);
        return r;
    }

    static int findCapPath() {
        for (int i = 0; i < M + N + 2; i++) {
            flow[i] = 0;
            visited[i] = false;
        }
        flow[0] = Integer.MAX_VALUE;
        prev[0] = null;
        while (true) {
            int maxCap = -1;
            int maxVertex = -1;

            for (int i = 0; i < M + N + 2; i++) {
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

            for (Edge e : graph[maxVertex]) {
                if (flow[e.dst] < Math.min(e.cap, maxCap)) {
                    flow[e.dst] = Math.min(e.cap, maxCap);
                    prev[e.dst] = e;
                }
            }
        }

        return flow[N + M + 1];
    }

    static class Edge {
        public final int src;
        public final int dst;
        public int cap;

        public Edge(int src, int dst, int cap) {
            this.src = src;
            this.dst = dst;
            this.cap = cap;
        }
    }
}
