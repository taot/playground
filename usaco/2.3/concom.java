/*
ID: libra_k1
LANG: JAVA
TASK: concom
*/
import java.io.*;
import java.util.*;

class concom {

    private static String task = "concom";

    static boolean[][] owns =  new boolean[101][101];
    static int[][] percent = new int[101][101];
    static int max;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        int n = Integer.parseInt(st.nextToken());
        max = 0;
        for (int i = 0; i < n; i++) {
            st = new StringTokenizer(f.readLine());
            int a = Integer.parseInt(st.nextToken());
            max = Math.max(a, max);
            int b = Integer.parseInt(st.nextToken());
            max = Math.max(b, max);
            int p = Integer.parseInt(st.nextToken());
            percent[a][b] = p;
        }

        for (int i = 1; i <= max; i++) {
            for (int j = 1; j <= max; j++) {
                // if (i == j) {
                //     continue;
                // }
                if (i == j || percent[i][j] > 50) {
                    owns[i][j] = true;
                    // System.out.println(i + " ==> " + j);
                }
            }
        }

        // for (int x = 1; x <= max; x++) {
        boolean changed;
        do {
            changed = false;
            for (int i = 1; i <= max; i++) {
                for (int j = 1; j <= max; j++) {
                    if (i == j) {
                        continue;
                    }
                    int sum = 0;
                    for (int k = 1; k <= max; k++) {
                        // if (k == i || k == j) {
                        //     continue;
                        // }
                        if (owns[i][k]) {
                            sum += percent[k][j];
                        }
                    }
                    // if (i == 34) {
                        // System.out.println(i + " ==> " + j + ": " + sum);
                    // }
                    if (sum > 50 && ! owns[i][j]) {
                        owns[i][j] = true;
                        changed = true;
                        // System.out.println(i + " ==> " + j + ": " + sum);
                    }
                }
            }
        } while (changed);

        // printPercent();

        for (int i = 1; i <= max; i++) {
            for (int j = 1; j <= max; j++) {
                if (i == j) {
                    continue;
                }
                if (owns[i][j]) {
                    out.println(i + " " + j);
                }
            }
        }

        out.close();
    }

    static void printPercent() {
        for (int i = 1; i <= max; i++) {
            System.out.println(i + ": ");
            for (int j = 1; j <= max; j++) {
                System.out.print(percent[i][j] + " ");
            }
            System.out.println();
        }
    }
}
