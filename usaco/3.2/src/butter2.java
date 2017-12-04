/*
ID: libra_k1
LANG: JAVA
TASK: butter
*/

import java.io.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.StringTokenizer;

class butter2 {

    private static String task = "butter";

    static int N, P, C;
    static List<Edge>[] graph;
    static int[] dists;
    static int[] cows;

    public static void main (String [] args) throws IOException {
        long start = System.currentTimeMillis();

        BufferedReader f = new BufferedReader(new FileReader(task + ".in9"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));

        // read input
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        P = Integer.parseInt(st.nextToken());
        C = Integer.parseInt(st.nextToken());
        cows = new int[N];
        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            cows[i] = Integer.parseInt(st.nextToken()) - 1;
        }

        graph = new List[P];
        dists = new int[P];
        for (int i = 0; i < P; i++) {
            graph[i] = new ArrayList<Edge>();
        }
        for (int i = 0; i < C; i++) {
            st = new StringTokenizer(f.readLine());
            int s = Integer.parseInt(st.nextToken()) - 1;
            int t = Integer.parseInt(st.nextToken()) - 1;
            int len = Integer.parseInt(st.nextToken());

            graph[s].add(new Edge(s, t, len));
            graph[t].add(new Edge(t, s, len));
        }

        // calculate
        int minSum = Integer.MAX_VALUE;
        for (int s = 0; s < P; s++) {
            dijkstra(s);
            int sum = 0;
            for (int i = 0; i < N; i++) {
                sum += dists[cows[i]];
            }
            if (sum < minSum) {
                minSum = sum;
            }
//            System.out.println(s + ": " + sum);
        }

        System.out.println(minSum);
        out.println(minSum);

        out.close();

        long duration = System.currentTimeMillis() - start;
        System.out.println("duration: " + duration + " ms");
    }

    static void dijkstra(int src) {
        FibHeap<Edge> heap = new FibHeap<Edge>(new Comparator<Edge>() {
            @Override
            public int compare(Edge o1, Edge o2) {
                return o1.length - o2.length;
            }
        });
        boolean[] visited = new boolean[P];
        int visitCount = 0;
        for (int i = 0; i < P; i++) {
            dists[i] = 0;
        }
        for (Edge e : graph[src]) {
            heap.insert(e);
            dists[e.t] = e.length;
        }
        visited[src] = true;
        visitCount++;

        while (visitCount < P) {
            Edge e;
            do {
                e = heap.extractMin();
            } while (visited[e.t]);
            int n = e.t;
            for (Edge f : graph[n]) {
                if (dists[f.t] == 0 && src != f.t || f.length + e.length < dists[f.t]) {
                    dists[f.t] = f.length + e.length;
                    heap.insert(new Edge(src, f.t, dists[f.t]));
                }
            }

            visited[n] = true;
            visitCount++;
        }
    }

    static class Edge {
        final int s;
        final int t;
        final int length;

        public Edge(int s, int t, int length) {
            this.s = s;
            this.t = t;
            this.length = length;
        }

        @Override
        public String toString() {
            return "Edge{" +
                    "s=" + s +
                    ", t=" + t +
                    ", length=" + length +
                    '}';
        }
    }

    static class FibNode<E> {
        E key;
        int degree;
        FibNode p;
        FibNode child;
        FibNode left;
        FibNode right;
        boolean mark;

        @Override
        public String toString() {
            return "Node{" +
                    "key=" + key +
                    ", degree=" + degree +
                    ", mark=" + mark +
                    '}';
        }
    }

    static class FibHeap<E> {

        public static double LOG_PHI = Math.log((1 + Math.sqrt(5)) / 2);

        FibNode<E> min;
        int n;
        boolean destroyed;
        Comparator<E> comparator;

        public FibHeap() {
            this(null);
        }

        public FibHeap(Comparator<E> comparator) {
            this.comparator = comparator;
            this.min = null;
            this.n = 0;
        }

        public void insert(E key) {
            check();

            FibNode<E> node = new FibNode<E>();
            node.key = key;
            node.degree = 0;
            node.p = null;
            node.child = null;
            node.mark = false;

            if (this.min == null) {
                this.min = node;
                node.left = node.right = node;
            } else {
                // insert node into root list
                insertIntoList(node, this.min);

                if (compare(node.key, this.min.key) < 0) {
                    this.min = node;
                }
            }

            this.n++;
        }

        public E getMin() {
            check();
            return this.min == null ? null : this.min.key;
        }

        public FibHeap union(FibHeap<E> h) {
            check();

            FibHeap<E> nh = new FibHeap<E>();

            if (this.min == null) {
                nh.min = h.min;
            } else if (h.min == null) {
                nh.min = this.min;
            } else {
                concat(this.min, h.min);
                nh.min = compare(this.min.key, h.min.key) < 0 ? this.min : h.min;
            }

            nh.n = this.n + h.n;
            this.destroyed = h.destroyed = true;

            return nh;
        }

        public E extractMin() {
            check();

            FibNode<E> z = this.min;
            if (z == null) {
                return null;
            }

            // add z's children to root list
            if (z.child != null) {
                FibNode n = z.child;
                do {
                    n.p = null;
                    n = n.right;
                } while (n != z.child);

                concat(this.min, z.child);
            }

            // remove z from root list
            if (z == z.right) {
                this.min = null;
            } else {
                this.min = z.right;
                z.right.left = z.left;
                z.left.right = z.right;
                consolidate();
            }

            this.n -= 1;

            return z.key;
        }

        private void consolidate() {
            int D = computeD();
            FibNode<E>[] A = new FibNode[D + 1];

            FibNode w = this.min;
            if (w == null) {
                return;
            }

            FibNode sentry = w;
            do {
                // for each node w in root list
                FibNode<E> x = w;
                int d = x.degree;
                while (A[d] != null) {
                    FibNode<E> y = A[d];
                    if (compare(x.key, y.key) > 0) {
                        FibNode tmp = x;
                        x = y;
                        y = tmp;
                    }

                    // link: remove y from root list and make y a child of x
                    if (y == sentry) {
                        sentry = y.right;
                    }
                    y.right.left = y.left;
                    y.left.right = y.right;
                    if (w == y) {
                        w = y.left;
                    }
                    insertIntoList(y, x.child);
                    if (x.child == null) {
                        x.child = y;
                    }
                    y.p = x;
                    x.degree++;
                    y.mark = false;

                    A[d] = null;
                    d++;
                }
                A[d] = x;

                w = w.right;
            } while (w != sentry);

            this.min = null;
            for (int i = 0; i <= D; i++) {
                if (A[i] != null) {
                    if (this.min == null) {
                        this.min = A[i];
                        this.min.right = this.min.left = this.min;
                    } else {
                        insertIntoList(A[i], this.min);
                        if (compare(A[i].key, this.min.key) < 0) {
                            this.min = A[i];
                        }
                    }
                }
            }
        }

        private int compare(E o1, E o2) {
            if (this.comparator != null) {
                return this.comparator.compare(o1, o2);
            }
            return ((Comparable<E>) o1).compareTo(o2);
        }

        private int computeD() {
            return (int) (Math.log(this.n) / LOG_PHI);
        }

        private void concat(FibNode n1, FibNode n2) {
            // concatenate two lists
            n1.right.left = n2.left;
            n2.left.right = n1.right;
            n1.right = n2;
            n2.left = n1;
        }

        private void insertIntoList(FibNode node, FibNode list) {
            if (list == null) {
                node.right = node.left = node;
            } else {
                node.right = list.right;
                list.right.left = node;
                node.left = list;
                list.right = node;
            }
        }

        private void check() {
            if (this.destroyed) {
                throw new IllegalStateException("this fibonacci heap is already destroyed");
            }
        }
    }
}
