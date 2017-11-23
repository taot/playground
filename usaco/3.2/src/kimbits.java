/*
ID: libra_k1
LANG: JAVA
TASK: kimbits
*/
import java.io.*;
import java.util.*;

class kimbits {

    private static String task = "kimbits";

    private static int N, L;
    private static long I;

    static final int MAX_N = 31;

    static int S[][] = new int[MAX_N + 1][MAX_N + 1];

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        L = Integer.parseInt(st.nextToken());
        I = Long.parseLong(st.nextToken());

        dp();

        System.out.println(S[3][2]);

//        int r = find(N, L, I);
        int r = k(N, L, I-1);
        String bits = toBits(r);
        System.out.println(bits);
        out.println(bits);

        out.close();
    }

    static int find(int n, int l, int i) {
        int r = 0;

        for (int k = n; k >= 1; k--) {
            int s = S[k-1][l];
            if (s < i) {
                r |= (1 << k);
                i -= s;
            } else {
                l--;
            }
        }

        return r;
    }

    static String toBits(int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < N; i++) {
            sb.append(n % 2 == 0 ? '0' : '1');
            n /= 2;
        }
        return sb.reverse().toString();
    }

    static int k(int n, int l, long i) {
        if (n == 0) {
            return 0;
        }
        int s = S[n-1][l];
        if (s <= i) {
            return exp2(n-1) + k(n-1, l-1, i-s);
        } else {
            return k(n-1, l, i);
        }
    }

    static int exp2(int n) {
        int p = 1;
        while (n > 0) {
            p *= 2;
            n--;
        }
        return p;
    }

    /**
     * Dynamic programming for s(n, l)
     */
    static void dp() {
        for (int i = 0; i <= MAX_N; i++) {
            S[i][0] = 1;
            S[0][i] = 1;
        }
        for (int n = 1; n <= MAX_N; n++) {
            for (int l = 1; l <= MAX_N; l++) {
                if (l > n) {
                    S[n][l] = S[n][n];
                } else {
                    S[n][l] = S[n-1][l] + S[n-1][l-1];
                }
            }
        }
    }
}