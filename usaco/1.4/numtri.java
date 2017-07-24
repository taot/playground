/*
ID: libra_k1
LANG: JAVA
TASK: numtri
*/
import java.io.*;
import java.util.*;

class numtri {

    private static String task = "numtri";

    static int NROWS;
    static int[][] tri;
    static int[][] max;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        NROWS = Integer.parseInt(st.nextToken());
        tri = new int[NROWS][NROWS];
        max = new int[NROWS][NROWS];
        for (int i = 0; i < NROWS; i++) {
            st = new StringTokenizer(f.readLine());
            for (int j = 0; j <= i; j++) {
                tri[i][j] = Integer.parseInt(st.nextToken());
            }
        }

        // for (int i = 0; i < NROWS; i++) {
        //     for (int j = 0; j <= i; j++) {
        //         System.out.print(tri[i][j] + " ");
        //     }
        //     System.out.println();
        // }

        DP();

        int m = -1;
        for (int i = 0; i < NROWS; i++) {
            if (m < max[NROWS-1][i]) {
                m = max[NROWS-1][i];
            }
        }
        // System.out.println("m = " + m);
        out.println(m);

        // for (int i = 0; i < NROWS; i++) {
        //     for (int j = 0; j <= i; j++) {
        //         System.out.print(max[i][j] + " ");
        //     }
        //     System.out.println();
        // }

        out.close();
    }

    static void DP() {
        max[0][0] = tri[0][0];
        for (int i = 1; i < NROWS; i++) {
            for (int j = 0; j <= i; j++) {
                int m;
                if (j == 0) {
                    m = max[i - 1][0];
                } else if (j == i) {
                    m = max[i-1][i-1];
                } else {
                    m = Math.max(max[i-1][j-1], max[i-1][j]);
                }
                max[i][j] = tri[i][j] + m;
            }
        }
    }
}
