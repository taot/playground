/*
ID: libra_k1
LANG: JAVA
TASK: subset
*/
import java.io.*;
import java.util.*;

class subset {

    private static String task = "subset";

    static int N;
    static int[][] mx = new int[40][820];

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());

        // mx[0][0] = mx[1][0] = mx[1][1] = 1;
        for (int i = 1; i <= N; i++) {
            mx[i][0] = 1;
        }
        for (int i = 2; i <= (N+1)*N/2; i++) {
            mx[1][i] = 0;
        }
        mx[1][1] = 1;

        for (int j = 1; j <= (N+1)*N/2; j++) {
            for (int i = 2; i <= N; i++) {
                if (j - i <= 0) {
                    mx[i][j] = mx[i-1][j];
                } else {
                    mx[i][j] = mx[i-1][j-i] + mx[i-1][j];
                }
            }
        }
        // for (int i = 1; i < N; i++) {
        //     for (int j = 1; j <= (i+1) * i / 2; j++) {
        //             mx[i][j] += mx[i-1][j-i];
        //     }
        // }

        // for (int i = 1; i <= N; i++) {
        //     for (int j = 0; j <= (N+1)*N/2; j++) {
        //         System.out.print(mx[i][j] + " ");
        //     }
        //     System.out.println();
        // }

        int c;
        if ((N+1)*N % 4 != 0) {
            c = 0;
        } else {
            c = mx[N][(N+1) * N / 4];
        }
        out.println(c);

        out.close();
    }


}
