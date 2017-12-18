/*
ID: libra_k1
LANG: JAVA
TASK: fence9
*/
import java.io.*;
import java.util.*;

class fence9 {

    private static String task = "fence9";

    static final double EPS = 1E-6;

    public static void main (String [] args) throws IOException {
        long start = System.currentTimeMillis();

        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        int n = Integer.parseInt(st.nextToken());
        int m = Integer.parseInt(st.nextToken());
        int p = Integer.parseInt(st.nextToken());

        Vector p1 = new Vector(n, m);
        Vector p2 = new Vector(0, 0);
        Vector p3 = new Vector(p, 0);

        double n2 = Math.max(n, p);

        int count = 0;
        for (int y = 1; y < m; y++) {
            Vector i1 = intersect(p1, p2, new Vector(0, y), new Vector(n2, y));
            Vector i2 = intersect(p1, p3, new Vector(0, y), new Vector(n2, y));
            int x1 = getLowerBound(i1.x);
            int x2 = getUpperBound(i2.x);
//            if (y == 88) {
//                System.out.println();
//            }
            if (x2 >= x1) {
                int c = x2 - x1 + 1;
//                System.out.println("y = " + y + ", c = " + c);
                count += c;
            }
        }

        System.out.println(count);
        out.println(count);

        out.close();

        System.out.println("Duration: " + (System.currentTimeMillis() - start) + " ms");
    }

    static int getUpperBound(double x) {
        int x1 = (int) Math.floor(x + EPS);
        int x2 = (int) Math.ceil(x - EPS);
        if (x1 == x2) {
            return x1 - 1;
        }
        return x1;
    }

    static int getLowerBound(double x) {
        int x1 = (int) Math.floor(x + EPS);
        int x2 = (int) Math.ceil(x - EPS);
        if (x1 == x2) {
            return x1 + 1;
        }
        return x2;
    }

    static Vector intersect(Vector a, Vector b, Vector c, Vector d) {
        double i = -1 * ((a.x - c.x) * (c.y - d.y) - (a.y - c.y) * (c.x - d.x)) / ((b.x - a.x) * (c.y - d.y) - (b.y - a.y) * (c.x - d.x));
        double j = -1 * ((a.x - c.x) * (b.y - a.y) - (a.y - c.y) * (b.x - a.x)) / ((c.x - d.x) * (b.y - a.y) - (c.y - d.y) * (b.x - a.x));
        double x = a.x + i * (b.x - a.x);
        double y = a.y + i * (b.y - a.y);
        return new Vector(x, y);
    }

    static class Vector {
        final double x;
        final double y;

        public Vector(double x, double y) {
            this.x = x;
            this.y = y;
        }

        public Vector subtr(Vector v) {
            return new Vector(this.x - v.x, this.y - v.y);
        }

        @Override
        public String toString() {
            return "Vector{" +
                    "x=" + x +
                    ", y=" + y +
                    '}';
        }
    }
}
