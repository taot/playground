/*
ID: libra_k1
LANG: JAVA
TASK: fence
*/
import java.io.*;
import java.util.*;

class fence {

    private static String task = "fence";

    final static int MAX_F = 1024;
    final static int MAX_N = 500;

    static int F;
    static int N = 0;

    static List<Edge>[] graph = new ArrayList[MAX_N];

    static int smallestOddDegreeNode = Integer.MAX_VALUE;
    static int smallestNode = Integer.MAX_VALUE;
    static int oddDegreeNodeCount = 0;

    static int[] circuit = new int[MAX_F + 1];
    static int circuitPos = 0;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        F = Integer.parseInt(st.nextToken());

        for (int i = 0; i < MAX_N; i++) {
            graph[i] = new ArrayList<Edge>();
        }

        for (int i = 0; i < F; i++) {
            st = new StringTokenizer(f.readLine());
            int s = Integer.parseInt(st.nextToken()) - 1;
            int t = Integer.parseInt(st.nextToken()) - 1;
            if (s > N) {
                N = s;
            }
            if (t > N) {
                N = t;
            }
            Edge e1 = new Edge(s, t);
            Edge e2 = new Edge(t, s);
            e1.other = e2;
            e2.other = e1;
            graph[s].add(e1);
            graph[t].add(e2);
        }

        for (int i = 0; i < MAX_N; i++) {
            Collections.sort(graph[i], new Comparator<Edge>() {
                @Override
                public int compare(Edge o1, Edge o2) {
                    return o1.t - o2.t;
                }
            });
        }

        findSmallestOddDegreeNode();
        if (oddDegreeNodeCount == 0) {
            findCircuit(smallestNode);
        } else {
            findCircuit(smallestOddDegreeNode);
        }

        for (int i = circuitPos - 1; i >= 0; i--) {
            out.println(circuit[i] + 1);
            System.out.println(circuit[i] + 1);
        }

        out.close();
    }

    static void findSmallestOddDegreeNode() {
        for (int i = 0; i < MAX_N; i++) {
            int size = graph[i].size();
            if (size > 0 && i < smallestNode) {
                smallestNode = i;
            }
            if (size % 2 == 1) {
                oddDegreeNodeCount++;
                if (i < smallestOddDegreeNode) {
                    smallestOddDegreeNode = i;
                }
            }
        }
        if (oddDegreeNodeCount != 0 && oddDegreeNodeCount != 2) {
            throw new RuntimeException("no euler path or circuit");
        }
    }

    static void findCircuit(int cur) {
        List<Edge> neigbors = graph[cur];
        if (neigbors == null || neigbors.isEmpty()) {
            circuit[circuitPos] = cur;
            circuitPos++;
            return;
        }
        while (! neigbors.isEmpty()) {
            Edge n = neigbors.get(0);
            removeEdge(n);
            findCircuit(n.t);
        }
        circuit[circuitPos] = cur;
        circuitPos++;
    }

    static void removeEdge(Edge e) {
        graph[e.s].remove(e);
        graph[e.t].remove(e.other);
    }

    static class Edge {
        final int s;
        final int t;
        Edge other;

        public Edge(int s, int t) {
            this.s = s;
            this.t = t;
        }

        @Override
        public String toString() {
            return "Edge{" +
                    "s=" + s +
                    ", t=" + t +
                    '}';
        }
    }
}
