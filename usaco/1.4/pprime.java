/*
ID: libra_k1
LANG: JAVA
TASK: pprime
*/
import java.io.*;
import java.util.*;

class pprime {

    private static String task = "pprime";

    static long A;
    static long B;

    static List<Long> list = new ArrayList<Long>();

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        A = Long.parseLong(st.nextToken());
        B = Long.parseLong(st.nextToken());

        // for (int i = A; i <= B; i++) {
        //     if (isPalindro(i) && isPrime(i)) {
        //         out.println(i);
        //     }
        // }

        // out.println(i1+i2);

        int lmt = numOfDigits(B);

        for (int len = 1; len <= lmt; len++) {
            int[] d = new int[len];
            recursive(d, 0);
        }

        Collections.sort(list);
        for (Long a : list) {
            // System.out.println(a);
            out.println(a);
        }

        out.close();
    }

    static int numOfDigits(long n) {
        int c = 0;
        while (n > 0) {
            n /= 10;
            c++;
        }
        return c;
    }

    // static void genPPrime(int len, boolean mid) {
    //     for ()
    // }

    static void recursive(int[] digits, int p) {
        int len = digits.length;
        int lmt;
        if (len % 2 == 0) {
            lmt = len / 2;
        } else {
            lmt = len / 2 + 1;
        }
        if (p >= lmt) {
            long n = toNum(digits);
            // System.out.println("n = " + n);
            // printDigits(digits);
            if (digits[0] != 0) {
                if (n >= A && n <= B && isPalindro(n) && isPrime(n)) {
                    list.add(n);
                }
            }
            return;
        }
        for (int i = 0; i <= 9; i++) {
            digits[p] = digits[len-1-p] = i;
            recursive(digits, p+1);
        }
    }

    static long toNum(int[] d) {
        long s = 0;
        for (int i = 0; i < d.length; i++) {
            s *= 10;
            s += d[i];
        }
        return s;
    }

    static void printDigits(int[] d) {
        for (int i = 0; i < d.length; i++) {
            System.out.print(d[i]);
        }
        System.out.println();
    }

    static boolean isPalindro(long n) {
        char[] s = String.valueOf(n).toCharArray();
        int len = s.length;
        for (int i = 0; i < len / 2; i++) {
            if (s[i] != s[len-i-1]) {
                return false;
            }
        }
        return true;
    }

    static boolean isPrime(long n) {
        long m = (long) Math.sqrt(n);
        for (int i = 2; i <= m; i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }
}
