/*
ID: libra_k1
LANG: JAVA
TASK: rockers
*/
import java.io.*;
import java.util.*;

class rockers {

    private static String task = "rockers";

    static int N;       // num of sons
    static int T;       // minutes of musics per disk
    static int M;       // num of disks

    static int W[];

    static int m[][][];
    static int F[][];

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        T = Integer.parseInt(st.nextToken());
        M = Integer.parseInt(st.nextToken());

        W = new int[N];
        st = new StringTokenizer(f.readLine());
        for (int i = 0; i < N; i++) {
            W[i] = Integer.parseInt(st.nextToken());
        }

        // calc m
        m = new int[T + 1][N + 1][N + 1];
        for (int i = 0; i < N + 1; i++) {
            for (int j = 0; j < N + 1; j++) {
                m[0][i][j] = 0;
            }
        }
        for (int w = 0; w <= T; w++) {
            m[w][0][0] = 0;
            for (int i = 0; i < N + 1; i++) {
                for (int j = 0; j < N + 1; j++) {
                    if (i >= j) {
                        m[w][i][j] = 0;
                        continue;
                    }
                    if (w - W[j - 1] < 0) {
                        m[w][i][j] = m[w][i][j - 1];
                    } else {
                        m[w][i][j] = Math.max(m[w][i][j - 1], m[w - W[j - 1]][i][j - 1] + 1);
                    }
                }
            }
        }

//        printArray(m);

        // calc F
        F = new int[M + 1][N + 1];
        for (int i = 0; i < M + 1; i++) {
            F[i][0] = 0;
        }
        for (int j = 0; j < N + 1; j++) {
            F[0][j] = 0;
        }
        for (int i = 1;  i < M + 1; i++) {
            for (int j = 1; j < N + 1; j++) {
                int max = 0;
                for (int k = 0; k <= j; k++) {
                    int x = F[i-1][k] + m[T][k][j];
                    if (x > max) {
                        max = x;
                    }
                }
                F[i][j] = max;
            }
        }

//        printArray(F);
        System.out.println(F[M][N]);
        out.println(F[M][N]);

        out.close();
    }

    static void printArray(int[] a) {
        for (int i = 0; i < a.length; i++) {
            System.out.print(a[i] + ' ');
        }
        System.out.println();
    }

    static void printArray(int[][] a) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                System.out.print(a[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

    static void printArray(int[][][] a) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                for (int k = 0; k < a[i][j].length; k++) {
                    System.out.print(a[i][j][k] + " ");
                }
                System.out.println();
            }
            System.out.println();
        }
        System.out.println();
    }
}
