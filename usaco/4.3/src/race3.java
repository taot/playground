/*
ID: libra_k1
LANG: JAVA
TASK: race3
*/
import java.io.*;
import java.util.*;

class race3 {

    private static String task = "race3";

    private static int N;
    private static List<Edge>[] directed;
    private static List<Edge>[] undirected;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        read(f);
        toUndirected();

        System.out.println(N);
        boolean[] visited = new boolean[N];
        List<Integer> unavoidables = new ArrayList<>();

        for (int i = 1; i < N - 1; i++) {
            clear(visited);
            bfs(directed, 0, i, -1, visited);
            if (! visited[N - 1]) {
                unavoidables.add(i);
            }
        }

        printList(System.out, unavoidables);
        printList(out, unavoidables);

        boolean[] visited2 = new boolean[N];
        List<Integer> splits = new ArrayList<>();
        for (int i = 1; i < N - 1; i++) {
            clear(visited);
            clear(visited2);
            bfs(directed, 0, i, -1, visited);
            bfs(directed, i, -1, i, visited2);
            boolean flag = visited[i] && visited2[i];
            for (int j = 0; j < N; j++) {
                flag = flag && ( i == j || (visited[j] ^ visited2[j]));
            }
            if (flag) {
                splits.add(i);
            }
        }

        printList(System.out, splits);
        printList(out, splits);

        out.close();
    }

    static void bfs(List<Edge>[] graph, int start, int outRemoved, int inRemoved, boolean[] visited) {
        Deque<Integer> q = new ArrayDeque<>();
        q.addFirst(start);
        visited[start] = true;
        Integer n;
        while ((n = q.pollLast()) != null) {
            if (outRemoved >= 0 && outRemoved == n) {
                continue;
            }
            for (Edge e : graph[n]) {
                if (visited[e.dst]) {
                    continue;
                }
                if (inRemoved >= 0 && e.dst == inRemoved) {
                    continue;
                }
                visited[e.dst] = true;
                q.addFirst(e.dst);
            }
        }
    }

    static void clear(boolean[] a) {
        for (int i = 0; i < a.length; i++) {
            a[i] = false;
        }
    }

    static void toUndirected() {
        undirected = new List[N];
        for (int i = 0; i < N; i++) {
            undirected[i] = new ArrayList<>();
        }
        for (int i = 0; i < N; i++) {
            for (Edge e : directed[i]) {
                undirected[i].add(new Edge(i, e.dst));
                undirected[e.dst].add(new Edge(e.dst, i));
            }
        }
    }

    static void read(BufferedReader f) throws IOException {
        List<List<Edge>> graph = new ArrayList<>();
        int src = 0;
        while (true) {
            String l = f.readLine();
            int[] nums = getLineData(l);
            if (nums[0] == -1) {
                break;
            }
            List<Edge> list = new ArrayList<>();
            graph.add(list);
            int i = 0;
            while (nums[i] >= 0) {
                list.add(new Edge(src, nums[i]));
                i++;
            }
            src++;
        }
        N = src;
        directed = graph.toArray(new List[0]);
    }

    static int[] getLineData(String s) {
        String[] parts = s.split(" ");
        int[] nums = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            nums[i] = Integer.parseInt(parts[i]);
        }
        return nums;
    }

    static void printList(PrintStream ps, List<Integer> list) {
        Collections.sort(list);
        ps.print(list.size());
        for (Integer i : list) {
            ps.print(" " + i);
        }
        ps.println();
    }

    static void printList(PrintWriter pw, List<Integer> list) {
        Collections.sort(list);
        pw.print(list.size());
        for (Integer i : list) {
            pw.print(" " + i);
        }
        pw.println();
    }

    static class Edge {
        final int src;
        final int dst;

        public Edge(int src, int dst) {
            this.src = src;
            this.dst = dst;
        }

        @Override
        public String toString() {
            return "Edge{" +
                    "src=" + src +
                    ", dst=" + dst +
                    '}';
        }
    }
}
