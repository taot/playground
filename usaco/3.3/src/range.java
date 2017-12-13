/*
ID: libra_k1
LANG: JAVA
TASK: range
*/
import java.io.*;
import java.util.*;

class range {

    private static String task = "range";

    static final int MAX_N = 250;
    static int N;
//    static char[][] fields;
    static char[][] array1;
    static char[][] array2;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
//        fields = new char[N][N];
        array1 = new char[N][N];
        array2 = new char[N][N];
        for (int i = 0; i < N; i++) {
            char[] a = f.readLine().toCharArray();
            System.arraycopy(a, 0, array1[i], 0, N);
        }

        char[][] a1 = array1;
        char[][] a2 = array2;
        for (int i = 2; i <= N; i++) {
            int count = 0;
            for (int j = 0; j < N - i + 1; j++) {
                for (int k = 0; k < N - i + 1; k++) {
                    a2[j][k] = '0';
                    if (a1[j][k] == '1' && a1[j][k+1] == '1' && a1[j+1][k] == '1' && a1[j+1][k+1] == '1') {
                        a2[j][k] = '1';
                        count++;
                    }
                }
            }
            if (count > 0) {
                System.out.println(i + " " + count);
                out.println(i + " " + count);
            }
            char[][] tmp = a1;
            a1 = a2;
            a2 = tmp;
        }

        out.close();
    }
}
