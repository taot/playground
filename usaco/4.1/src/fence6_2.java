/*
ID: libra_k1
LANG: JAVA
TASK: fence6
*/
import java.io.*;
import java.util.*;

class fence6_2 {

    private static String task = "fence6";

    static final int MAX_N = 100;
    static List<Edge>[] graph = new ArrayList[MAX_N];
    static int[][] graph2 = new int[MAX_N][MAX_N];
    static int[][] dists = new int[MAX_N][MAX_N];
    static boolean[][] updated = new boolean[MAX_N][MAX_N];

    static Map<Integer, Integer> nodeIdMap = new HashMap<>();
    static int curNodeNo = 0;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        read(f);

        printGraph();
        createGraph2();
        floyd();
        System.out.println();
        printGraph2(dists);


        out.close();
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
                    if (k == i || k == j ) {
                        continue;
                    }
                    if (dists[i][k] > 0 && dists[k][j] > 0) {
                        int s = dists[i][k] + dists[k][j];
                        if (dists[i][j] < 0 || dists[i][j] > s) {
//                            dists[i][j] = dists[j][i] = s;
//                            updated[i][j] = updated[j][i] = true;
                            dists[i][j] = s;
                            updated[i][j] = true;
                        }
                    }
                }
            }
        }
    }

    static void createGraph2() {
        for (int i = 0; i < curNodeNo; i++) {
            for (Edge e : graph[i]) {
                graph2[e.src][e.dst] = graph2[e.dst][e.src] = e.len;
            }
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
            int nodeId1 = readNodeId(st, N1, segNo);
            st = new StringTokenizer(f.readLine());
            int nodeId2 = readNodeId(st, N2, segNo);

            int nodeNo1 = getNodeNo(nodeId1);
            int nodeNo2 = getNodeNo(nodeId2);

            graph[nodeNo1].add(new Edge(nodeNo1, nodeNo2, len, segNo));
            graph[nodeNo2].add(new Edge(nodeNo2, nodeNo1, len, segNo));
        }
    }

    static int getNodeNo(int nodeId) {
        Integer no = nodeIdMap.get(nodeId);
        if (no == null) {
            no = curNodeNo;
            curNodeNo++;
            nodeIdMap.put(nodeId, no);
        }
        return no;
    }

    static int readNodeId(StringTokenizer st, int n, int segNo) {
        int id = (1 << segNo);
        for (int i = 0; i < n; i++) {
            int s = Integer.parseInt(st.nextToken());
            id |= (1 << s);
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
