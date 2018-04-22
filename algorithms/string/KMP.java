public class KMP {

    public static void main(String[] args) {
        char[] s = "abababcdaee".toCharArray();
        char[] p = "eab".toCharArray();
        int[] t = table(p);
        printIntArray(t);
        int r = kmp(s, p);
        System.out.println(r);
    }

    static int kmp(char[] s, char[] p) {
        int[] t = table(p);
        int i = 0;
        int j = 0;
        while (i < s.length && j < p.length) {
            if (s[i] == p[j]) {
                j++;
                i++;
            } else if (j > 0) {
                j = t[j-1];
            } else {
                j = 0;
                i++;
            }
        }
        if (j >= p.length) {
            return i - j;
        }
        return -1;
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
