/*
ID: libra_k1
LANG: JAVA
TASK: ariprog
*/
import java.io.*;
import java.util.*;

class ariprog {

    private static String task = "ariprog";

    static int N;
    static int M;
    static PrintWriter out;
    static int count = 0;

    static BitSet isBisquare = new BitSet();
    static BitSet notBisqare = new BitSet();

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        st = new StringTokenizer(f.readLine());
        M = Integer.parseInt(st.nextToken());

        search();
        if (count == 0) {
            out.println("NONE");
        }

        out.close();
    }

    private static void search() {
        int upper = 2 * M * M;
        for (int b = 1; b <= upper; b++) {
            for (int a = 0; a <= upper - (N - 1) * b; a++) {
                if (is_bisquare_prog(a, b)) {
                    out.println(a + " " + b);
                    count++;
                }
            }
        }
    }

    private static boolean is_bisquare_prog(int a, int b) {
        for (int i = N - 1; i >= 0; i--) {
            if (! is_bisquare(a + b * i)) {
                return false;
            }
        }
        return true;
    }

    private static boolean is_bisquare(int n) {
        if (isBisquare.get(n)) {
            return true;
        }
        if (notBisqare.get(n)) {
            return false;
        }
        int ceil = (int) Math.min(M, Math.sqrt(n));
        for (int i = ceil; i >= 0; i--) {
            int x = (int) Math.sqrt(n - i * i);
            if (x > M) {
                break;
            }
            if (x * x + i * i == n) {
                isBisquare.set(n);
                return true;
            }
        }
        notBisqare.set(n);
        return false;
    }
}
