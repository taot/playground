/*
ID: libra_k1
LANG: JAVA
TASK: job
*/
import java.io.*;
import java.util.*;

class job {

    private static String task = "job";

    static int N, M1, M2;
    static int[] ptA, ptB;      // processing times
    static int[] ftA, ftB;        // finish times

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        M1 = Integer.parseInt(st.nextToken());
        M2 = Integer.parseInt(st.nextToken());
        ptA = new int[M1];
        ptB = new int[M2];
        ftA = new int[M1];
        ftB = new int[M2];

        st = new StringTokenizer(f.readLine());
        for (int i = 0; i < M1; i++) {
            if (! st.hasMoreTokens()) {
                st = new StringTokenizer(f.readLine());
            }
            ptA[i] = Integer.parseInt(st.nextToken());
            ftA[i] = 0;
        }
        for (int i = 0; i < M2; i++) {
            if (! st.hasMoreTokens()) {
                st = new StringTokenizer(f.readLine());
            }
            ptB[i] = Integer.parseInt(st.nextToken());
            ftB[i] = 0;
        }

        // calculate
        for (int i = 0; i < N; i++) {
            int min_T1 = Integer.MAX_VALUE;
            int min_j1 = -1;
            for (int j = 0; j < M1; j++) {
                if (ftA[j] + ptA[j] < min_T1) {
                    min_j1 = j;
                    min_T1 = ftA[j] + ptA[j];
                }
            }
            ftA[min_j1] = min_T1;
        }

        for (int i = 0; i < N; i++) {
            int min_T2 = Integer.MAX_VALUE;
            int min_j2 = -1;
            for (int j = 0; j < M2; j++) {
                for (int k = 0; k < M1; k++) {
                    if (Math.max(ftB[j], ftA[k]) + ptB[j] < min_T2) {
                        min_j2 = j;
                        min_T2 = Math.max(ftB[j], ftA[k]) + ptB[j];
                    }
                }
            }
            ftB[min_j2] = min_T2;
        }

        // print
        int t1 = getMax(ftA);
        int t2 = getMax(ftB);

        System.out.println(t1 + " " + t2);
        out.println(t1 + " " + t2);
        printArr(ftA);
        printArr(ftB);

        out.close();
    }

    static int getMax(int[] a) {
        int m = Integer.MIN_VALUE;
        for (int i : a) {
            if (i > m) {
                m = i;
            }
        }
        return m;
    }

    static void printArr(int[] a) {
        for (int i : a) {
            System.out.print(i + " ");
        }
        System.out.println();
    }
}
