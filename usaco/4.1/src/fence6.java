/*
ID: libra_k1
LANG: JAVA
TASK: fence6
*/
import java.io.*;
import java.util.*;

class fence6 {

    private static String task = "fence6";

    static final int MAX_N = 100;
    static List<Edge>[] graph = new ArrayList[MAX_N];
    static Map<NodeId, Integer> nodeIdMap = new HashMap<>();
    static int curNodeNo = 0;
    static boolean[] visited = new boolean[MAX_N];
    static int minPeri = Integer.MAX_VALUE;
    static int[][] graph2 = new int[MAX_N][MAX_N];
    static int[][] dists = new int[MAX_N][MAX_N];

    public static void main (String [] args) throws IOException {
        long startTs = System.currentTimeMillis();
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        read(f);

        createGraph2();
        floyd();
//        printGraph();
//        System.out.println();
//        printGraph2(dists);

        for (int i = 0; i < curNodeNo; i++) {
//            System.out.println("i = " + i);
            sortEdges(i);
            visited[i] = true;
            dfs(i, i, 0, -1);
            visited[i] = false;
        }

        System.out.println(minPeri);
        out.println(minPeri);

        out.close();

        System.out.println("Duration: " + (System.currentTimeMillis() - startTs) + " ms");
    }

    static void dfs(int start, int node, int len, int lastSegNo) {
        for (Edge e : graph[node]) {
            if (e.segNo == lastSegNo) {
                continue;
            }
            if (start == e.dst) {
                if (len + e.len < minPeri) {
                    minPeri = len + e.len;
                }
                continue;
            }
            if (visited[e.dst]) {
                continue;
            }
            if (len + e.len > minPeri) {
                continue;
            }
            visited[e.dst] = true;
            dfs(start, e.dst, len + e.len, e.segNo);
            visited[e.dst] = false;
        }
    }

    static void createGraph2() {
        for (int i = 0; i < curNodeNo; i++) {
            for (Edge e : graph[i]) {
                graph2[e.src][e.dst] = graph2[e.dst][e.src] = e.len;
            }
        }
    }

    static void floyd() {
        for (int i = 0; i < curNodeNo; i++) {
            for (int j = 0; j < curNodeNo; j++) {
                if (graph2[i][j] > 0) {
                    dists[i][j] = graph2[i][j];
                } else {
                    dists[i][j] = -1;
                }
            }
        }
        for (int k = 0; k < curNodeNo; k++) {
            for (int i = 0; i < curNodeNo; i++) {
                for (int j = 0; j < curNodeNo; j++) {
                    if (k == i || k == j || i == j) {
                        continue;
                    }
                    if (dists[i][k] > 0 && dists[k][j] > 0) {
                        int s = dists[i][k] + dists[k][j];
                        if (dists[i][j] < 0 || dists[i][j] > s) {
                            dists[i][j] = dists[j][i] = s;
                        }
                    }
                }
            }
        }
    }

    static void sortEdges(int start) {
        for (int i = 0; i < curNodeNo; i++) {
            Collections.sort(graph[i], new Comparator<Edge>() {
                @Override
                public int compare(Edge o1, Edge o2) {
                    return dists[start][o1.dst] - dists[start][o2.dst];
                }
            });
        }
    }

    static void printGraph2(int[][] mat) {
        for (int i = 0; i < curNodeNo; i++) {
            for (int j = 0; j < curNodeNo; j++) {
                System.out.print(mat[i][j] + " ");
            }
            System.out.println();
        }
    }

    static void printGraph() {
        for (int i = 0; i < curNodeNo; i++) {
            System.out.print(i + ":");
            for (Edge e : graph[i]) {
                assert(e.src == i);
                System.out.print(String.format(" %d(%d,%d)", e.dst, e.segNo, e.len));
            }
            System.out.println();
        }
    }

    static void read(BufferedReader f) throws IOException {
        for (int i = 0; i < MAX_N; i++) {
            graph[i] = new ArrayList<>();
        }

        StringTokenizer st = new StringTokenizer(f.readLine());
        int N = Integer.parseInt(st.nextToken());
        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            int segNo = Integer.parseInt(st.nextToken());
            int len = Integer.parseInt(st.nextToken());
            int N1 = Integer.parseInt(st.nextToken());
            int N2 = Integer.parseInt(st.nextToken());

            st = new StringTokenizer(f.readLine());
            NodeId nodeId1 = readNodeId(st, N1, segNo);
            st = new StringTokenizer(f.readLine());
            NodeId nodeId2 = readNodeId(st, N2, segNo);

            int nodeNo1 = getNodeNo(nodeId1);
            int nodeNo2 = getNodeNo(nodeId2);

            graph[nodeNo1].add(new Edge(nodeNo1, nodeNo2, len, segNo));
            graph[nodeNo2].add(new Edge(nodeNo2, nodeNo1, len, segNo));
        }
    }

    static int getNodeNo(NodeId nodeId) {
        Integer no = nodeIdMap.get(nodeId);
        if (no == null) {
            no = curNodeNo;
            curNodeNo++;
            nodeIdMap.put(nodeId, no);
        }
        return no;
    }

    static NodeId readNodeId(StringTokenizer st, int n, int segNo) {
//        int id = (1 << segNo);
        NodeId id = new NodeId();
        id.add(segNo);
        for (int i = 0; i < n; i++) {
            int s = Integer.parseInt(st.nextToken());
//            id |= (1 << s);
            id.add(s);
        }
        return id;
    }

    static class Path {
        final int dst;
        final int len;
        final int lastSegNo;

        public Path(int dst, int len, int lastSegNo) {
            this.dst = dst;
            this.len = len;
            this.lastSegNo = lastSegNo;
        }
    }

    static class NodeId {
        long high = 0L;
        long low = 0L;

//        public NodeId(long high, long low) {
//            this.high = high;
//            this.low = low;
//        }

        public void add(int n) {
            if (n > 64) {
                n -= 64;
                high |= (1L << n);
            } else {
                low |= (1L << n);
            }
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            NodeId nodeId = (NodeId) o;
            return high == nodeId.high &&
                    low == nodeId.low;
        }

        @Override
        public int hashCode() {

            return Objects.hash(high, low);
        }
    }

    static class Edge {
        final int src;
        final int dst;
        final int len;
        final int segNo;

        public Edge(int src, int dst, int len, int segNo) {
            this.src = src;
            this.dst = dst;
            this.len = len;
            this.segNo = segNo;
        }

        @Override
        public String toString() {
            return "Edge{" +
                    "src=" + src +
                    ", dst=" + dst +
                    ", len=" + len +
                    ", segNo=" + segNo +
                    '}';
        }
    }
}
