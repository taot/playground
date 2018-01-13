/*
ID: libra_k1
LANG: JAVA
TASK: milk6
*/
import java.io.*;
import java.util.*;

class milk6 {

    private static String task = "milk6";

    static PrintWriter out;

    static int N, M;
    static long[][] cost_graph;
    static long[][] weight_graph;
    static Edge[] edges;

    static int[] pre;
    static long[] flow;
    static boolean[] visited;

    static List<Set<Integer>> minCuts = new ArrayList<>();

    public static void main (String [] args) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(task + ".in"));
        out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        read(reader);

        boolean[][] exclude = new boolean[N][N];
        long max_flow = maxFlow(exclude);

        solve(max_flow);

        out.close();
    }

    static void solve(long max_flow) {
        boolean[][] minCut = new boolean[N][N];
        long mf = max_flow;
//        for (int i = 0; i < N; i++) {
//            for (int j = 0; j < N; j++) {
        for (Edge e : edges) {
            int i = e.src;
            int j = e.dst;
            if (minCut[i][j]) {
                continue;
            }
            if (cost_graph[i][j] == 0) {
                continue;
            }
            minCut[i][j] = true;
            long f = maxFlow(minCut);
            if (mf - f != weight_graph[i][j]) {
                minCut[i][j] = false;
            } else {
                mf = f;
            }
        }
        int T = 0;
        int C = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (minCut[i][j]) {
                    C += cost_graph[i][j];
                }
            }
        }
        for (Edge e : edges) {
            if (minCut[e.src][e.dst]) {
                T++;
            }
        }

        System.out.println(C + " " + T);
        out.println(C + " " + T);
        for (Edge e : edges) {
            if (minCut[e.src][e.dst]) {
                System.out.println(e.idx + 1);
                out.println(e.idx + 1);
            }
        }
    }

//    static void solveExp(int max_flow) {
//        for (int i = 0; i < N; i++) {
//            for (int j = 0; j < N; j++) {
//                if (i == j || cost_graph[i][j] == 0) {
//                    continue;
//                }
//                boolean[][] exclude = new boolean[N][N];
//                exclude[i][j] = true;
//                int mf = maxFlow(exclude);
//                if (max_flow - mf != edges[i].cap) {
//                    continue;
//                }
//                if (mf > 0) {
//                    for (int j = 0; j < M; j++) {
//                        if (mf == 0) {
//                            break;
//                        }
//                        if (j == i) {
//                            continue;
//                        }
//                        exclude.add(j);
//                        int f = maxFlow(exclude);
//                        if (mf - f != edges[j].cap) {
//                            exclude.remove(j);
//                        } else {
//                            mf = f;
//                        }
//                    }
//                }
//
//                minCuts.add(exclude);
//            }
//        }
//    }

    static long maxFlow(boolean[][] exclude) {
        long total = 0;
        long[][] g = copyGraph();
        while (true) {
            long cap = findCapPath(g, exclude);
            if (cap == 0) {
                break;
            }
            total += cap;
            int dst = N - 1;
            int src = pre[N-1];
            while (src >= 0) {
                g[src][dst] -= cap;
                g[dst][src] += cap;
                dst = src;
                src = pre[dst];
            }
        }
        return total;
    }

    static Edge getReverseEdge(List<Edge>[] g, Edge e) {
        for (Edge i : g[e.dst]) {
            if (e.src == i.dst) {
                return i;
            }
        }
        Edge r = new Edge(-1, e.dst, e.src, e.cost, 0);
        g[e.dst].add(r);
        return r;
    }

    static long findCapPath(long[][] g, boolean[][] exclude) {
        for (int i = 0; i < N; i++) {
            visited[i] = false;
            flow[i] = 0;
            pre[i] = -1;
        }
        flow[0] = Long.MAX_VALUE;

        while (true) {
            // find i that's unvisited and with max flow
            int max_i = -1;
            long max_flow = 0;
            for (int i = 0; i < N; i++) {
                if (! visited[i] && flow[i] > max_flow) {
                    max_i = i;
                    max_flow = flow[i];
                }
            }

            if (max_i < 0) {
                break;
            }

            // update neighbors' flows
//            for (Edge e : g[max_i]) {
            for (int i = 0; i < N; i++) {
                if (i == max_i || exclude[max_i][i] || visited[i]) {
                    continue;
                }
                if (flow[i] < Math.min(flow[max_i], g[max_i][i])) {
                    flow[i] = Math.min(flow[max_i], g[max_i][i]);
                    pre[i] = max_i;
                }
            }
            visited[max_i] = true;
        }

        return flow[N-1];
    }

    static void read(BufferedReader reader) throws IOException {
        StringTokenizer st = new StringTokenizer(reader.readLine());
        N = Integer.parseInt(st.nextToken());
        M = Integer.parseInt(st.nextToken());
        cost_graph = new long[N][N];
        weight_graph = new long[N][N];
        edges = new Edge[M];
        pre = new int[N];
        flow = new long[N];
        visited = new boolean[N];

        for (int i = 0; i < M; i++) {
            st = new StringTokenizer(reader.readLine());
            int src = Integer.parseInt(st.nextToken()) - 1;
            int dst = Integer.parseInt(st.nextToken()) - 1;
            int cost = Integer.parseInt(st.nextToken());
            Edge e = new Edge(i, src, dst, cost, cost * M + 1);
            edges[i] = e;
            cost_graph[src][dst] += cost;
            weight_graph[src][dst] += cost * M + 1;
        }
    }

//    static List<Edge>[] copyGraph() {
//        List<Edge>[] g = new List[N];
//        for (int i = 0; i < N; i++) {
//            g[i] = new ArrayList<>();
//        }
//        for (int i = 0; i < N; i++) {
//            for (Edge e : graph[i]) {
//                Edge e2 = new Edge(e.idx, e.src, e.dst, e.cost, e.cap);
//                g[i].add(e2);
//            }
//        }
//        return g;
//    }

    static long[][] copyGraph() {
        long[][] g = new long[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                g[i][j] = weight_graph[i][j];
            }
        }
        return g;
    }

    static class Edge {
        final int idx;
        final int src;
        final int dst;
        final int cost;
        int cap;

        public Edge(int idx, int src, int dst, int cost, int cap) {
            this.idx = idx;
            this.src = src;
            this.dst = dst;
            this.cost = cost;
            this.cap = cap;
        }

        @Override
        public String toString() {
            return "Edge{" +
                    "idx=" + idx +
                    ", src=" + src +
                    ", dst=" + dst +
                    ", cost=" + cost +
                    ", cap=" + cap +
                    '}';
        }
    }
}
