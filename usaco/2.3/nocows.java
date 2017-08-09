/*
ID: libra_k1
LANG: JAVA
TASK: nocows
*/
import java.io.*;
import java.util.*;

class nocows {

    private static String task = "nocows";

    static int N;
    static int K;

    static int[] P;
    static int[][] PK;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        K = Integer.parseInt(st.nextToken());

        P = new int[N+1];
        PK = new int[N+1][K+1];

        // calc1();
        calc2();

        // printP();
        // printPK();

        // out.println(PK[N][K]);
        out.println(PK[N][K] % 9901);

        out.close();
    }

    static void calc2() {
        for (int i = 0; i <= N; i++) {
            PK[i][0] = 0;
        }
        for (int i = 0; i <= K; i++) {
            PK[0][i] = 0;
        }
        PK[1][1] = 1;

        for (int i = 2; i <= N; i++) {
            for (int j = 1; j <= K; j++) {

                // if (i == 165 && j == 65) {
                //     System.out.println("something");
                // }

                // if (i < j * 2 -1 || i > pow(2, j) - 1) {
                //     PK[i][j] = 0;
                //     continue;
                // }

                int s0 = 0;
                for (int x = 1; x <= i-2; x++) {
                    int s1 = 0;
                    for (int k = 0; k <= j-2; k++) {
                        s1 += PK[i-x-1][k];
                        s1 = s1 % 9901;
                    }
                    s0 += PK[x][j-1] * s1;
                    s0 = s0 % 9901;
                }

                int s2 = 0;
                for (int x = 1; x <= i-2; x++) {
                    s2 += PK[x][j-1] * PK[i-x-1][j-1];
                    s2 = s2 % 9901;
                }

                if (i == 165 && j == 65) {
                    System.out.println(String.format("s0 = %d, s2 = %d", s0, s2));
                }

                int s = s0 * 2 + s2;
                PK[i][j] = s % 9901;
            }
        }
    }

    static void calc1() {
        P[0] = P[1] = 1;
        P[2] = 0;
        for (int i = 3; i <= N; i++) {
            if (i % 2 == 0) {
                P[i] = 0;
                continue;
            }
            int sum = 0;
            for (int j = 1; j <= i-2; j++) {
                sum += P[j] * P[i-j-1];
            }
            P[i] = sum;
        }

        for (int i = 0; i <= N; i++) {
            PK[i][0] = 0;
        }
        for (int i = 0; i <= K; i++) {
            PK[0][i] = 0;
        }
        PK[1][1] = 1;

        for (int i = 2; i <= N; i++) {
            for (int j = 1; j <= K; j++) {


                if (i < j * 2 -1 || i > pow(2, j) - 1) {
                    // System.out.println(String.format("i = %d, j = %d, pow(2,j) = %d", i, j, pow(2,j)));
                    PK[i][j] = 0;
                    continue;
                }
                // System.out.println(String.format("i = %d, j = %d\n", i, j));

                int s1 = 0;
                for (int x = 1; x <= i-2; x++) {
                    s1 += PK[x][j-1] * P[i-x-1];
                }

                int s2 = 0;
                for (int x = 1; x <= i-2; x++) {
                    s2+= PK[x][j-1] * PK[i-x-1][j-1];
                }

                // System.out.println(String.format("i = %d, j = %d, s1 = %d, s2 = %d", i, j, s1, s2));

                PK[i][j] = 2 * s1 - s2;
            }
        }
    }

    static void printP() {
        for (int i = 0; i <= N; i++) {
            System.out.println(i + " " + P[i]);
        }
    }

    static void printPK() {
        for (int i = 0; i <= N; i++) {
            System.out.print("i = " + i + ": ");
            for (int j = 0; j <= K; j++) {
                System.out.print(PK[i][j] + " ");
            }
            System.out.println();
        }
    }

    static int pow(int m, int n) {
        int p = 1;
        for (int i = 0; i < n; i++) {
            p *= m;
        }
        return p;
    }
}
