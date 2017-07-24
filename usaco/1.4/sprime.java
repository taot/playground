/*
ID: libra_k1
LANG: JAVA
TASK: sprime
*/
import java.io.*;
import java.util.*;

class sprime {

    private static String task = "sprime";

    static int LEN;

    static List<List<Integer>> list = new ArrayList<>();

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        LEN = Integer.parseInt(st.nextToken());

        for (int i = 0; i < LEN; i++) {
            list.add(new ArrayList<Integer>());
        }

        DP();

        for (int n : list.get(LEN-1)) {
            out.println(n);
        }

        // for (int i = lower + 1; i < upper; i++) {
        //     if (isSPrime(i)) {
        //         out.println(i);
        //     }
        // }

        out.close();
    }

    static void DP() {
        List<Integer> l = list.get(0);
        l.add(2);
        l.add(3);
        l.add(5);
        l.add(7);
        for (int i = 1; i < LEN; i++) {
            List<Integer> l0 = list.get(i-1);
            List<Integer> l1 = list.get(i);
            for (int n : l0) {
                n *= 10;
                for (int j = 1; j <= 9; j+=2) {
                    int x = n + j;
                    if (isPrime(x)) {
                        l1.add(x);
                    }
                }
            }
        }
    }

    static int getLower(int n) {
        int x = 1;
        for (int i = 1; i < n; i++) {
            x *= 10;
        }
        return x;
    }

    static boolean isSPrime(int n) {
        while (n > 0) {
            if (! isPrime(n)) {
                return false;
            }
            n /= 10;
        }
        return true;
    }

    static boolean isPrime(int n) {
        if (n == 1) {
            return false;
        }
        if (n == 2) {
            return true;
        }
        int m = (int) Math.sqrt(n);
        for (int i = 2; i <= m; i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }
}
