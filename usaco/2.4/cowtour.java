/*
ID: libra_k1
LANG: JAVA
TASK: cowtour
*/
import java.io.*;
import java.util.*;

class cowtour {

    private static String task = "cowtour";
    static int N;
    static Point[] points;
    static double[][] D;
    static int nComps;
    static double[] diameters;
    static double[] maxD;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        points = new Point[N];
        D = new double[N][N];

        // read points
        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            int x = Integer.parseInt(st.nextToken());
            int y = Integer.parseInt(st.nextToken());
            points[i] = new Point(x, y);
        }

        // read adj
        for (int i = 0; i < N; i++) {
            char[] chars = f.readLine().toCharArray();
            for (int j = 0; j < N; j++) {
                if (chars[j] == '1') {
                    D[i][j] = points[i].dist(points[j]);
                } else {
                    D[i][j] = -1;
                }
            }
        }

        findComponents();
        System.out.println("nComps: " + nComps);
        diameters = new double[nComps];
        System.out.println("diameters:");
        for (int i = 0; i < nComps; i++) {
            diameters[i] = calcDiameter(i, i);
            System.out.print(diameters[i] + " ");
        }
        System.out.println();

        maxD = new double[N];
        System.out.println("maxD");
        for (int i = 0; i < N; i++) {
            double m = 0;
            for (int j = 0; j < N; j++) {
                if (D[i][j] > m) {
                    m = D[i][j];
                }
            }
            maxD[i] = m;
            System.out.print(m + " ");
        }
        System.out.println();

        double minDia = Double.MAX_VALUE;
        for (int i = 0; i < nComps - 1; i++) {
            double m1 = diameters[i];
            for (int j = i+1; j < nComps; j++) {
                double m2 = diameters[j];
                for (int a = 0; a < N-1; a++) {
                    Point pa = points[a];
                    if (pa.comp != i) {
                        continue;
                    }
                    for (int b = a+1; b < N; b++) {
                        Point pb = points[b];
                        if (pb.comp != j) {
                            continue;
                        }
                        double m = max(m1, m2, pa.dist(pb) + maxD[a] + maxD[b]);
                        if (m < minDia) {
                            minDia = m;
                        }
                    }
                }
            }
        }

        out.println(String.format("%.6f", minDia));

        out.close();
    }

    static double max(double d1, double d2, double d3) {
        double max = d1;
        if (d2 > max) {
            max = d2;
        }
        if (d3 > max) {
            max = d3;
        }
        return max;
    }

    static double calcDiameter(int c1, int c2) {
        double[][] dist = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                dist[i][j] = D[i][j];
            }
        }
        // printD(dist);
        for (int k = 0; k < N; k++) {
            if (points[k].comp != c1 && points[k].comp != c2) {
                continue;
            }
            for (int i = 0; i < N; i++) {
                if (points[i].comp != c1 && points[i].comp != c2) {
                    continue;
                }
                for (int j = 0; j < N; j++) {
                    if (points[j].comp != c1 && points[j].comp != c2 || i == j) {
                        continue;
                    }
                    double d;
                    if (dist[i][k] < 0 || dist[k][j] < 0) {
                        d = -1;
                    } else {
                        d = dist[i][k] + dist[k][j];
                    }
                    if (dist[i][j] < 0 || (d > 0 && dist[i][j] > d)) {
                        dist[i][j] = dist[j][i] = d;
                        D[i][j] = D[j][i] = d;
                    }
                }
            }
        }

        double max = 0;
        for (int i = 0; i < N; i++) {
            if (points[i].comp != c1 && points[i].comp != c2) {
                continue;
            }
            for (int j = 0; j < N; j++) {
                double d = dist[i][j];
                if (d > max) {
                    max = d;
                }
            }
        }
        return max;
    }

    static void printComponents() {
        for (int i = 0; i < N; i++) {
            System.out.println(i + ": " + points[i].comp);
        }
    }

    static void findComponents() {
        int comp = 0;

        BitSet visited = new BitSet();
        int unvisited = findUnvisited(visited);
        while (unvisited >= 0) {
            Integer x = unvisited;
            Deque<Integer> q = new ArrayDeque<>();
            Point p = points[x];
            q.addLast(x);
            visited.set(x);
            p.comp = comp;

            while ((x = q.pollFirst()) != null) {
                p = points[x];
                for (int i = 0; i < N; i++) {
                    if (D[x][i] > 0.1 && ! visited.get(i)) {
                        Point p1 = points[i];
                        q.addLast(i);
                        visited.set(i);
                        p1.comp = comp;
                    }
                }
            }

            comp++;
            unvisited = findUnvisited(visited);
        }
        nComps = comp;
    }

    static int findUnvisited(BitSet set) {
        for (int i = 0; i < N; i++) {
            if (! set.get(i)) {
                return i;
            }
        }
        return -1;
    }

    static void printD(double[][] d) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                System.out.print(String.format("%.6f ", d[i][j]));
            }
            System.out.println();
        }
    }

    static double sqr(int a) {
        return a * a;
    }

    static class Point {
        public int x;
        public int y;
        // public int idx;
        public int comp = -1;
        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }
        public double dist(Point p) {
            return Math.sqrt(sqr(x - p.x) + sqr(y - p.y));
        }
    }
}
