/*
ID: libra_k1
LANG: JAVA
TASK: milk6
*/
import java.io.*;
import java.util.*;

class milk6_1 {

    private static String task = "milk6";

    static PrintWriter out;

    static int N, M;
    static List<Edge>[] graph;
    static Edge[] edges;

    static Edge[] pre;
    static int[] flow;
    static boolean[] visited;

    static List<Set<Integer>> minCuts = new ArrayList<>();

    public static void main (String [] args) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(task + ".in5"));
        out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        read(reader);

        int max_flow = maxFlow(Collections.emptySet());

        solve(max_flow);

        out.close();
    }

    static void solve(int max_flow) {
        Set<Integer> minCut = new TreeSet<>();
        int mf = max_flow;
        for (int i = 0; i < M; i++) {
            minCut.add(i);
            int f = maxFlow(minCut);
            if (mf - f != edges[i].cap) {
                minCut.remove(i);
            } else {
                mf = f;
            }
        }

        int T = minCut.size();
        int C = 0;
        for (int i : minCut) {
            C += edges[i].cost;
        }
        System.out.println(C + " " + T);
        out.println(C + " " + T);
        for (int i : minCut) {
            System.out.println(i + 1);
            out.println(i + 1);
        }
    }

    static void solveExp(int max_flow) {
        for (int i = 0; i < M; i++) {
            TreeSet<Integer> exclude = new TreeSet<>();
            exclude.add(i);
            int mf = maxFlow(exclude);
            if (max_flow - mf != edges[i].cap) {
                continue;
            }
            if (mf > 0) {
                for (int j = 0; j < M; j++) {
                    if (mf == 0) {
                        break;
                    }
                    if (j == i) {
                        continue;
                    }
                    exclude.add(j);
                    int f = maxFlow(exclude);
                    if (mf - f != edges[j].cap) {
                        exclude.remove(j);
                    } else {
                        mf = f;
                    }
                }
            }

            minCuts.add(exclude);
        }
    }

    static int maxFlow(Set<Integer> exclude) {
        int total = 0;
        List<Edge>[] g = copyGraph();
        while (true) {
            int cap = findCapPath(g, exclude);
            if (cap == 0) {
                break;
            }
            total += cap;
            Edge e = pre[N-1];
            while (e != null) {
                e.cap -= cap;
                Edge r = getReverseEdge(g, e);
                r.cap += cap;
                e = pre[e.src];
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

    static int findCapPath(List<Edge>[] g, Set<Integer> exclude) {
        for (int i = 0; i < N; i++) {
            visited[i] = false;
            flow[i] = 0;
            pre[i] = null;
        }
        flow[0] = Integer.MAX_VALUE;

        while (true) {
            // find i that's unvisited and with max flow
            int max_i = -1;
            int max_flow = 0;
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
            for (Edge e : g[max_i]) {
                if (exclude.contains(e.idx)) {
                    continue;
                }
                if (flow[e.dst] < Math.min(flow[max_i], e.cap)) {
                    flow[e.dst] = Math.min(flow[max_i], e.cap);
                    pre[e.dst] = e;
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
        graph = new List[N];
        edges = new Edge[M];
        pre = new Edge[N];
        flow = new int[N];
        visited = new boolean[N];

        for (int i = 0; i < N; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int i = 0; i < M; i++) {
            st = new StringTokenizer(reader.readLine());
            int src = Integer.parseInt(st.nextToken()) - 1;
            int dst = Integer.parseInt(st.nextToken()) - 1;
            int cost = Integer.parseInt(st.nextToken());
            Edge e = new Edge(i, src, dst, cost, cost * M + 1);
            edges[i] = e;
            graph[src].add(e);
        }
    }

    static List<Edge>[] copyGraph() {
        List<Edge>[] g = new List[N];
        for (int i = 0; i < N; i++) {
            g[i] = new ArrayList<>();
        }
        for (int i = 0; i < N; i++) {
            for (Edge e : graph[i]) {
                Edge e2 = new Edge(e.idx, e.src, e.dst, e.cost, e.cap);
                g[i].add(e2);
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
