/*
ID: libra_k1
LANG: JAVA
TASK: camelot
*/

import java.io.*;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.StringTokenizer;

class camelot1 {

    private static String task = "camelot";

    final static int MAX_R = 30, MAX_C = 26;
    final static int MAX_VALUE = 10000;

    static int R, C;
    static int kingX, kingY;
    static int knightCount = 0;
    static int[] knightsX = new int[MAX_C * MAX_R];
    static int[] knightsY = new int[MAX_C * MAX_R];

//    static int[][] dists = new int[MAX_R][MAX_C];
//    static int[][] dists2 = new int[MAX_R][MAX_C];
    static boolean[][] visited = new boolean[MAX_R][MAX_C];
    static int[][] distsAll = new int[MAX_R * MAX_C][MAX_R * MAX_C];

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));

        readInputs(f);

        long start = System.currentTimeMillis();

        bfsAll();

        System.out.println("after bfsAll(): " + (System.currentTimeMillis() - start) + " ms");

        int minSum = Integer.MAX_VALUE;

        for (int x = 0; x < C; x++) {
            for (int y = 0; y < R; y++) {
                // gather at x, y

                //** case 1 - no pick up
                // knights' move
                int sum0 = 0;
                for (int i = 0; i < knightCount; i++) {
                    sum0 += distsAll [encode(x, y)] [encode(knightsX[i], knightsY[i])];
                }
                // king's move
                int sum1 = sum0 + Math.max(Math.abs(kingX - x), Math.abs(kingY - y));
                if (sum1 < minSum) {
                    minSum = sum1;
                }

                //** case 2 - pick up
                for (int dx = -2; dx <= 2; dx++) {
                    for (int dy = -2; dy <= 2; dy++) {
                        if (kingX + dx < 0 || kingX + dx >= C || kingY + dy < 0 || kingY + dy >= R) {
                            continue;
                        }

                        int x1 = kingX + dx;
                        int y1 = kingY + dy;

                        for (int i = 0; i < knightCount; i++) {
                            // pick up at x1, y1 by knight i
                            int sum2 = sum0 - distsAll [encode(x, y)] [encode(knightsX[i], knightsY[i])];
                            sum2 += distsAll [encode(x1, y1)] [encode(knightsX[i], knightsY[i])];
                            sum2 += distsAll [encode(x1, y1)] [encode(x, y)];
                            sum2 += Math.max(Math.abs(kingX - x1), Math.abs(kingY - y1));

                            if (sum2 < minSum) {
                                minSum = sum2;
                            }
                        }
                    }
                }
            }
        }

        System.out.println("after calc(): " + (System.currentTimeMillis() - start) + " ms");

        System.out.println(minSum);
        out.println(minSum);

        out.close();
    }

    static int encode(int x, int y) {
        return y * C + x;
    }

    static void bfsAll() {
        int[][] ds = new int[MAX_R][MAX_C];
        for (int x = 0; x < C; x++) {
            for (int y = 0; y < R; y++) {
                bfs(x, y, ds);
                int i = encode(x, y);
                for (int x1 = 0; x1 < C; x1++) {
                    for (int y1 = 0; y1 < R; y1++) {
                        int j = encode(x1, y1);
                        distsAll[i][j] = distsAll[j][i] = ds[y1][x1];
                    }
                }
            }
        }
    }

    static void bfs(int x, int y, int[][] ds) {
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                visited[i][j] = false;
                ds[i][j] = MAX_VALUE;
            }
        }
        visited[y][x] = true;
        ds[y][x] = 0;
        Deque<Point> q = new ArrayDeque<Point>();
        q.addFirst(new Point(x, y, 0));
        Point p = null;
        Point[] adjs = new Point[8];
        while ((p = q.pollLast()) != null) {
            int count = getAdjacents(p.x, p.y, adjs);
            for (int i = 0; i < count; i++) {
                Point p1 = adjs[i];
                if (! visited[p1.y][p1.x]) {
                    ds[p1.y][p1.x] = p1.d = p.d + 1;
                    q.addFirst(p1);
                    visited[p1.y][p1.x] = true;
                }
            }
        }
    }

    static int getAdjacents(int x, int y, Point[] adjs) {
        int c = 0;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                if (Math.abs(i * j) == 2 && x + i >= 0 && x + i < C && y + j >= 0 && y + j < R) {
                    adjs[c] = new Point(x + i, y + j);
                    c++;
                }
            }
        }
        return c;
    }

    static void readInputs(BufferedReader f) throws IOException {
        StringTokenizer st = new StringTokenizer(f.readLine());
        R = Integer.parseInt(st.nextToken());
        C = Integer.parseInt(st.nextToken());
        st = new StringTokenizer(f.readLine());
        kingX = st.nextToken().charAt(0) - 'A';
        kingY = Integer.parseInt(st.nextToken()) - 1;
        String line = null;
        while ((line = f.readLine()) != null) {
            st = new StringTokenizer(line);
            while (st.hasMoreTokens()) {
                knightsX[knightCount] = st.nextToken().charAt(0) - 'A';
                knightsY[knightCount] = Integer.parseInt(st.nextToken()) - 1;
                knightCount++;
            }
        }
    }

    static class Point {
        public final int x;
        public final int y;
        public int d = MAX_VALUE;

        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }

        public Point(int x, int y, int d) {
            this(x, y);
            this.d = d;
        }

        @Override
        public String toString() {
            return "Point{" +
                    "x=" + x +
                    ", y=" + y +
                    '}';
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Point point = (Point) o;

            if (x != point.x) return false;
            return y == point.y;
        }

        @Override
        public int hashCode() {
            int result = x;
            result = 31 * result + y;
            return result;
        }
    }
}
