/*
ID: libra_k1
LANG: JAVA
TASK: palsquare
*/
import java.io.*;
import java.util.*;

class palsquare {

    private static String task = "palsquare";

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        int base = Integer.parseInt(st.nextToken());

        for (int n = 1; n <= 300; n++) {
            int sqr = n * n;
            char[] sqrchars = fromDecimal(sqr, base);
            if (isPalindrome(sqrchars)) {
                char[] chars = fromDecimal(n, base);
                out.print(new String(chars));
                out.print(' ');
                out.println(new String(sqrchars));
            }
        }

        out.close();
    }

    private static int toDecimal(char[] chars, int base) {
        int n = 0;
        for (int i = 0; i < chars.length; i++) {
            n *= base;
            n += (chars[i] - 'A' + 10);
        }
        return n;
    }

    private static char[] fromDecimal(int num, int base) {
        char[] chars = new char[100];
        int n = 0;
        while (num > 0) {
            int x = num % base;
            char c;
            if (x > 9) {
                c = (char)('A' + x - 10);
            } else {
                c = (char)('0' + x);
            }
            chars[n] = c;
            n++;
            num /= base;
        }
        chars = reverse(chars, n);
        return chars;
    }

    private static char[] reverse(char[] chars, int n) {
        char[] result = new char[n];
        for (int i = 0; i < n; i++) {
            result[i] = chars[n - i - 1];
        }
        return result;
    }

    private static boolean isPalindrome(char[] chars) {
        int n = chars.length;
        for (int i = 0; i < n / 2; i++) {
            if (chars[i] != chars[n - i - 1]) {
                return false;
            }
        }
        return true;
    }
}
