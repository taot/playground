/*
ID: libra_k1
LANG: JAVA
TASK: ditch
*/
import java.io.*;
import java.util.*;

class ditch {

    private static String task = "ditch";

    static int N, M;

    static final int MAX_N = 200;       // # of edges
    static final int MAX_M = 200;       // # of vertices

    static List<Edge>[] graph = new List[MAX_M];
    static Edge[] prev = new Edge[MAX_M];
    static int[] flow = new int[MAX_M];
    static boolean[] visited = new boolean[MAX_M];


    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        M = Integer.parseInt(st.nextToken());

        for (int i = 0; i < M; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            int s = Integer.parseInt(st.nextToken()) - 1;
            int e = Integer.parseInt(st.nextToken()) - 1;
            int c = Integer.parseInt(st.nextToken());
            graph[s].add(new Edge(s, e, c));
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
//            List<Integer> list = new ArrayList<>();
            while (cur != 0) {
//                list.add(cur);
                Edge pre = prev[cur];
                pre.cap -= maxFlow;
                Edge r = getReverseEdge(pre);
                r.cap += maxFlow;
                cur = pre.src;
            }
//            System.out.print(maxFlow + ": 0");
//            for (int i = list.size() - 1; i >= 0; i--) {
//                System.out.print(" -> " + list.get(i));
//            }
//            System.out.println();
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
        for (int i = 0; i < M; i++) {
            flow[i] = 0;
            visited[i] = false;
        }
        flow[0] = Integer.MAX_VALUE;
        prev[0] = null;
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

            for (Edge e : graph[maxVertex]) {
                if (flow[e.dst] < Math.min(e.cap, maxCap)) {
                    flow[e.dst] = Math.min(e.cap, maxCap);
                    prev[e.dst] = e;
                }
            }
        }

        return flow[M-1];
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
