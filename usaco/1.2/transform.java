/*
ID: libra_k1
LANG: JAVA
TASK: transform
*/
import java.io.*;
import java.util.*;

class transform {

    private static String task = "transform";

    private static int N;
    private static int H;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        H = N / 2;
        char[][] p1 = read(f);
        char[][] p2 = read(f);

        // System.out.println(equal(p1, p2));
        // print(p1);
        // char[][] q = new t4().run(p1);
        // print(q);

        t[] list = new t[] {
            new t1(),
            new t2(),
            new t3(),
            new t4(),
            new t51(),
            new t52(),
            new t53()
        };

        t found = null;
        for (int i = 0; i < list.length; i++) {
            char[][] q = list[i].run(p1);
            if (equal(q, p2)) {
                found = list[i];
                break;
            }
        }

        if (found != null) {
            out.print(found.getId());
        } else {
            out.print(7);
        }
        out.println();

        // out.println(i1+i2);
        out.close();
    }

    private static void print(char[][] p) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                System.out.print(p[i][j]);
            }
            System.out.println();
        }
    }

    private static char[][] read(BufferedReader f) throws IOException {
        char[][] p = new char[N][N];
        for (int i = 0; i < N; i++) {
            String line = f.readLine();
            char[] chars = line.toCharArray();
            p[i] = chars;
        }
        return p;
    }

    private static boolean equal(char[][] p1, char[][] p2) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (p1[i][j] != p2[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    private static interface t {
        int getId();
        char[][] run(char[][] p);
    }

    // 90 degree
    private static class t1 implements t {
        public int getId() {
            return 1;
        }
        public char[][] run(char[][] p) {
            char[][] q = new char[N][N];
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    q[j][N - 1 - i] = p[i][j];
                }
            }
            return q;
        }
    }

    // 180 degree
    private static class t2 implements t {
        public int getId() {
            return 2;
        }
        public char[][] run(char[][] p) {
            t tr = new t1();
            p = tr.run(p);
            p = tr.run(p);
            return p;
        }
    }

    // 270 degree
    private static class t3 implements t {
        public int getId() {
            return 3;
        }
        public char[][] run(char[][] p) {
            t tr = new t1();
            p = tr.run(p);
            p = tr.run(p);
            p = tr.run(p);
            return p;
        }
    }

    // reflect
    private static class t4 implements t {
        public int getId() {
            return 4;
        }
        public char[][] run(char[][] p) {
            char[][] q = new char[N][N];
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    q[i][j] = p[i][N - 1 - j];
                }
            }
            return q;
        }
    }

    private static class t51 implements t {
        public int getId() {
            return 5;
        }
        public char[][] run(char[][] p) {
            p = new t4().run(p);
            p = new t1().run(p);
            return p;
        }
    }

    private static class t52 implements t {
        public int getId() {
            return 5;
        }
        public char[][] run(char[][] p) {
            p = new t4().run(p);
            p = new t2().run(p);
            return p;
        }
    }

    private static class t53 implements t {
        public int getId() {
            return 5;
        }
        public char[][] run(char[][] p) {
            p = new t4().run(p);
            p = new t3().run(p);
            return p;
        }
    }

    private static class t6 implements t {
        public int getId() {
            return 6;
        }
        public char[][] run(char[][] p) {
            return p;
        }
    }
}
