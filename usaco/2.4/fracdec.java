/*
ID: libra_k1
LANG: JAVA
TASK: fracdec
*/
import java.io.*;
import java.util.*;

class fracdec {

    private static String task = "fracdec";
    static long N;
    static long D;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Long.parseLong(st.nextToken());
        D = Long.parseLong(st.nextToken());

        long a = N / D;
        long r = N % D;

        // find 2^i and 5^j and x
        long x = D;
        long i = 0;
        long j = 0;
        while (x % 2 == 0) {
            i++;
            x /= 2;
        }
        while (x % 5 == 0) {
            j++;
            x /= 5;
        }
        long y = findY(x);
        System.out.println(String.format("a = %d, r = %d, i = %d, j = %d, y = %d", a, r, i, j, y));

        long m = Math.max(i, j);
        // long N1 = r * exp(2, m-i) * exp(5, m-j) * exp(10, m) * y / x;
        // long D1 = exp(2, m) * exp(5, m) * y;
        long N1 = r * exp(2, m-i) * exp(5, m-j) * exp(10, m) * y / x;
        long D1 = exp(2, m) * exp(5, m) * y;

        System.out.println(String.format("m = %d, N1 = %d, D1 = %d", m, N1, D1));

        long nrep = N1 / D1;
        long rep = N1 % D1 / exp(10, m);

        StringBuilder sb = new StringBuilder();
        sb.append(a);
        sb.append(".");
        if (nrep == 0 && rep == 0) {
            sb.append("0");
        }

            String fmt = "%" + nDec(y) + "d";
            sb.append(pad0(nrep, nDec(y) - 1));
            // sb.append(nrep);
        // }
        if (rep != 0) {
            sb.append("(" + rep + ")");
        }
        // System.out.println(String.format("%d.%d(%d)", a, nrep, rep));
        System.out.println(sb.toString());

        out.close();
    }

    static String pad0(long a, int n) {
        String s = String.valueOf(a);
        while (s.length() < n) {
            s = "0" + s;
        }
        return s;
    }

    static int nDec(long n) {
        int c = 0;
        while (n > 0) {
            n /= 10;
            c++;
        }
        return c;
    }

    static long findY(long x) {
        long y = 9;
        while (y % x != 0) {
            y = y * 10 + 9;
        }
        return y;
    }

    static long exp(long a, long b) {
        long p = 1;
        for (long i = 0; i < b; i++) {
            p *= a;
        }
        return p;
    }

    static long findExp(long r, long b) {
        long e = 0;
        while (r % b == 0) {
            e++;
            r /= b;
        }
        return e;
    }
}
