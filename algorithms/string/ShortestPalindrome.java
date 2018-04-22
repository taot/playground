/**
 * LeetCode 214. Shortest Palindrome
 */
public class ShortestPalindrome {

    public static void main(String[] args) {
        char[] s = "abcd".toCharArray();
        find(s);
    }

    static void find(char[] s) {
        char[] rev = reverse(s);
        int[] t = table(s);

        int i = 0;
        int j = 0;
        int max_j = -1;
        while (i < rev.length && j < s.length) {
            if (rev[i] == s[j]) {
                if (j > max_j) {
                    max_j = j;
                }
                i++;
                j++;
            } else if (j > 0) {
                j = t[j-1];
            } else {
                j = 0;
                i++;
            }
        }
        char[] res = new char[s.length - max_j - 1];
        for (int k = 0; k < s.length - max_j - 1; k++) {
            res[k] = s[s.length - k - 1];
        }
        printCharArray(res);
    }

    static char[] reverse(char[] s) {
        char[] t = new char[s.length];
        for (int i = 0; i < s.length; i++) {
            t[i] = s[s.length - i - 1];
        }
        return t;
    }

    static int[] table(char[] p) {
        int[] t = new int[p.length];
        t[0] = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] == p[ t[i-1] ]) {
                t[i] = t[i-1] + 1;
            } else {
                t[i] = 0;
            }
        }
        return t;
    }

    static void printIntArray(int[] a) {
        for (int n : a) {
            System.out.print(n + " ");
        }
        System.out.println();
    }

    static void printCharArray(char[] cs) {
        for (char c : cs) {
            System.out.print(c);
        }
        System.out.println();
    }
}
