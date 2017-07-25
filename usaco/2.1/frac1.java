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

    static List<Frac> list = new ArrayList<>();

    public static void main (String [] args) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(reader.readLine());
        N = Integer.parseInt(st.nextToken());

        list.clear();
        for (int i = 1; i <= N; i++) {
            for (int j = 0; j < i; j++) {
                if (gcd(i, j) == 1) {
                    list.add(new Frac(j, i));
                }
            }
        }

        Collections.sort(list);
        for (Frac f : list) {
            out.println(f);
        }
        out.println(new Frac(1, 1));

        out.close();
    }

    static int gcd(int m , int n) {
        while (n != 0) {
            int x = m % n;
            m = n;
            n = x;
        }
        return m;
    }

    static class Frac implements Comparable<Frac> {
        public int n;
        public int d;

        public Frac(int n, int d) {
            this.n = n;
            this.d = d;
        }

        public String toString() {
            return n + "/" + d;
        }

        public int compareTo(Frac f) {
            return this.n * f.d - f.n * this.d;
        }
    }
}
