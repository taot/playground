/*
ID: libra_k1
LANG: JAVA
TASK: fence9
*/

import java.io.*;
import java.util.StringTokenizer;

class fence9_1 {

    private static String task = "fence9";

    public static void main (String [] args) throws IOException {
        long start = System.currentTimeMillis();

        BufferedReader f = new BufferedReader(new FileReader(task + ".in6"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        int n = Integer.parseInt(st.nextToken());
        int m = Integer.parseInt(st.nextToken());
        int p = Integer.parseInt(st.nextToken());

        Vector p1 = new Vector(n, m);
        Vector p2 = new Vector(0, 0);
        Vector p3 = new Vector(p, 0);
        Vector mid = getMid(p1, p2, p3);

        int count = 0;
        for (int y = 1; y < m; y++) {
            int x1 = 1;
            while (! isInside(p1, p2, p3, mid, new Vector(x1, y)) && x1 <= Math.max(n, p)) {
                x1++;
            }
            int x2 = Math.max(n, p);
            while (! isInside(p1, p2, p3, mid, new Vector(x2, y)) && x2 >= 0) {
                x2--;
            }
            if (x2 >= x1) {
                int c = x2 - x1 + 1;
                System.out.println("y = " + y + ", c = " + c);
                count += c;
            }
        }

        System.out.println(count);
        out.println(count);

        out.close();

        System.out.println("Duration: " + (System.currentTimeMillis() - start) + " ms");
    }

    static Vector intersect(Vector a, Vector b, Vector c, Vector d) {
        double i = -1 * ((a.x - c.x) * (c.y - d.y) - (a.y - c.y) * (c.x - d.x)) / ((b.x - a.x) * (c.y - d.y) - (b.y - a.y) * (c.x - d.x));
        double j = -1 * ((a.x - c.x) * (b.y - a.y) - (a.y - c.y) * (b.x - a.x)) / ((c.x - d.x) * (b.y - a.y) - (c.y - d.y) * (b.x - a.x));
        double x = a.x + i * (b.x - a.x);
        double y = a.y + j * (b.y - a.y);
        return new Vector(x, y);
    }

    static Vector getMid(Vector p1, Vector p2, Vector p3) {
        return new Vector((p1.x + p2.x + p3.x) / 3, (p1.y + p2.y + p3.y) / 3);
    }

    static boolean isInside(Vector p1, Vector p2, Vector p3, Vector mid, Vector v) {
        boolean b1 = sameSide(p1, p2, mid, v);
        boolean b2 = sameSide(p2, p3, mid, v);
        boolean b3 = sameSide(p3, p1, mid, v);
        return b1 && b2 && b3;
    }

    static boolean sameSide(Vector p1, Vector p2, Vector u, Vector v) {
        int s1 = getSide(p2.subtr(p1), u.subtr(p1));
        int s2 = getSide(p2.subtr(p1), v.subtr(p1));
        return s1 * s2 > 0;
    }

    static int getSide(Vector u, Vector v) {
        double d = u.x * v.y - u.y * v.x;
        if (d == 0) {
            return 0;
        } else if (d > 0) {
            return 1;
        } else {
            return -1;
        }
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
