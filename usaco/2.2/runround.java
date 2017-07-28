/*
ID: libra_k1
LANG: JAVA
TASK: runround
*/
import java.io.*;
import java.util.*;

class runround {

    private static String task = "runround";

    static int M;
    static int[] digits;
    static boolean[] visited;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        M = Integer.parseInt(st.nextToken());

        int n = M;
        while (true) {
            n++;
            digits = toDigits(n);
            if (! possible()) {
                continue;
            }
            // System.out.println(n);
            visited = new boolean[digits.length];
            if (walk()) {
                out.println(n);
                break;
            }
        }

        out.close();
    }

    static boolean possible() {
        int[] counts = new int[9];
        for (int x : digits) {
            if (x == 0) {
                return false;
            }
            counts[x-1]++;
        }
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] > 1) {
                return false;
            }
        }
        return true;
    }

    static boolean walk() {
        int s = 0;
        // System.out.println(digits.length);
        int n = digits[s];
        visited[0] = true;
        while (true) {
            s = (s + n) % digits.length;
            n = digits[s];

            if (visited[s] || completed()) {
                break;
            }
            visited[s] = true;
        }

        return (s == 0 && completed());
    }

    static boolean completed() {
        for (int i = 0; i < visited.length; i++) {
            if (! visited[i]) {
                return false;
            }
        }
        return true;
    }

    static int[] toDigits(int n) {
        List<Integer> list = new ArrayList<>();
        while (n > 0) {
            int x = n % 10;
            n = n / 10;
            list.add(x);
        }
        Collections.reverse(list);
        // System.out.println(list);
        int[] a = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            a[i] = list.get(i);
        }
        return a;
    }
}
