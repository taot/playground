/*
ID: libra_k1
LANG: JAVA
TASK: frac1
*/
import java.io.*;
import java.util.*;

class frac1 {

    private static String task = "frac1";

    static int N;
    static PrintWriter out;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());

        out.println("0/1");
        gen(0, 1, 1, 1);
        out.println("1/1");

        out.close();
    }

    static void gen(int n1, int d1, int n2, int d2) {
        if (d1 + d2 > N) {
            return;
        }
        gen(n1, d1, n1+n2, d1+d2);
        out.println(String.format("%d/%d", n1+n2, d1+d2));
        gen(n1+n2, d1+d2, n2, d2);
    }
}
