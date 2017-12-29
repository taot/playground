/*
ID: libra_k1
LANG: JAVA
TASK: fence6
*/
import java.io.*;
import java.util.*;

class fence6_1 {

    private static String task = "fence6";

    static final int MAX_N = 100;
    static List<Edge>[] graph = new ArrayList[MAX_N];

    static Map<Integer, Integer> nodeIdMap = new HashMap<>();
    static int curNodeNo = 0;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        read(f);

        printGraph();

        int minPeri = Integer.MAX_VALUE;
        for (int n = 0; n < curNodeNo; n++) {
            int c = bfs(n);
            if (c < minPeri) {
                minPeri = c;
            }
        }

        System.out.println(minPeri);

        out.close();
    }

    static int bfs(int start) {
        clearEdgeVisit();
        Deque<Path> q = new ArrayDeque<>();
        boolean[] visited = new boolean[MAX_N];
        q.addFirst(new Path(start, 0, -1));
        visited[start] = true;
        Path p;
        int minPeri = Integer.MAX_VALUE;

        while ((p = q.pollLast()) != null) {
            for (Edge e : graph[p.dst]) {
//                if (e.visited) {
//                    continue;
//                }
                if (e.dst == start && p.lastSegNo != e.segNo) {
                    if (e.len + p.len < minPeri) {
                        minPeri = e.len + p.len;
                    }
                }
//                e.visited = e.same.visited = true;
//                e.visited = true;
                if (! visited[e.dst]) {
                    q.addFirst(new Path(e.dst, p.len + e.len, e.segNo));
                    visited[e.dst] = true;
                }
            }
        }

        return minPeri;
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

            Edge e1 = new Edge(nodeNo1, nodeNo2, len, segNo);
            Edge e2 = new Edge(nodeNo2, nodeNo1, len, segNo);
//            e1.same = e2;
//            e2.same = e1;
            graph[nodeNo1].add(e1);
            graph[nodeNo2].add(e2);
        }
    }

    static void clearEdgeVisit() {
        for (int i = 0; i < curNodeNo; i++) {
            for (Edge e : graph[i]) {
                e.visited = false;
            }
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

        @Override
        public String toString() {
            return "Path{" +
                    "dst=" + dst +
                    ", len=" + len +
                    ", lastSegNo=" + lastSegNo +
                    '}';
        }
    }

    static class Edge {
        final int src;
        final int dst;
        final int len;
        final int segNo;
//        Edge same;
        boolean visited;

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
                    ", visited=" + visited +
                    '}';
        }
    }
}
