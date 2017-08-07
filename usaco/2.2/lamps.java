/*
ID: libra_k1
LANG: JAVA
TASK: lamps
*/
import java.io.*;
import java.util.*;

class lamps {

    private static String task = "lamps";

    static int N;
    static int C;
    static boolean[] ON;
    static boolean[] OFF;

    static boolean[][] states;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());

        N = Integer.parseInt(st.nextToken());
        st = new StringTokenizer(f.readLine());
        C = Integer.parseInt(st.nextToken());
        ON = new boolean[N];
        OFF = new boolean[N];

        st = new StringTokenizer(f.readLine());
        int x = Integer.parseInt(st.nextToken());
        while (x > 0) {
            ON[x-1] = true;
            x = Integer.parseInt(st.nextToken());
        }

        st = new StringTokenizer(f.readLine());
        x = Integer.parseInt(st.nextToken());
        while (x > 0) {
            OFF[x-1] = true;
            x = Integer.parseInt(st.nextToken());
        }

        boolean[] init = new boolean[N];
        for (int i = 0; i < N; i++) {
            init[i] = true;
        }
        states = new boolean[16][N];
        int count = 0;
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                for (int c = 0; c < 2; c++) {
                    for (int d = 0; d < 2; d++) {
                        boolean[] s = new boolean[N];
                        for (int i = 0; i < N; i++) {
                            s[i] = init[i];
                        }
                        for (int i = 0; i < N; i++) {
                            if (a == 1) {
                                s[i] = ! s[i];
                            }
                            if (b == 1 && i % 2 == 1) {
                                s[i] = ! s[i];
                            }
                            if (c == 1 && i % 2 == 0) {
                                s[i] = ! s[i];
                            }
                            if (d == 1 && i % 3 == 0) {
                                s[i] = ! s[i];
                            }
                        }
                        // int t = times(a, b, c, d);
                        if (! exists(s, count) && satisfy(s)) {
                            int m = C;
                            boolean add = false;
                            if (m % 2 == 0) {
                                if (m > 4) {
                                    m = 4;
                                }
                            } else {
                                if (m > 4) {
                                    m = 3;
                                }
                            }
                            int t = times(a, b, c, d);
                            while (m >= 0) {
                                if (m == t) {
                                    add = true;
                                }
                                m -= 2;
                            }
                            if (add) {
                                states[count] = s;
                                count++;
                            }
                        }
                    }
                }
            }
        }

        if (count == 0) {
            out.println("IMPOSSIBLE");
        } else {
            Arrays.sort(states, 0, count, new Comparator<boolean[]>() {
                public int compare(boolean[] a1, boolean[] a2) {
                    for (int i = 0; i < a1.length; i++) {
                        if (a1[i] != a2[i]) {
                            return a1[i] ? 1 : -1;
                        }
                    }
                    return 0;
                }
            });

            for (int i = 0; i < count; i++) {
                for (int j = 0; j < N; j++) {
                    out.print(states[i][j] ? 1 : 0);
                }
                out.println();
            }
        }

        out.close();
    }

    static int times(int a, int b, int c, int d) {
        int n = 0;
        if (a > 0) {
            n++;
        }
        if (b > 0) {
            n++;
        }
        if (c > 0) {
            n++;
        }
        if (d > 0) {
            n++;
        }
        return n;
    }

    static boolean exists(boolean[] s, int count) {
        for (int i = 0; i < count; i++) {
            boolean equal = true;
            for (int j = 0; j < N; j++) {
                if (s[j] != states[i][j]) {
                    equal = false;
                }
            }
            if (equal) {
                return true;
            }
        }
        return false;
    }

    static boolean satisfy(boolean[] s) {
        for (int i = 0; i < N; i++) {
            if (ON[i] && ! s[i]) {
                return false;
            }
            if (OFF[i] && s[i]) {
                return false;
            }
        }
        return true;
    }
}
