/*
ID: libra_k1
LANG: JAVA
TASK: hamming
*/
import java.io.*;
import java.util.*;

class hamming {

    private static String task = "hamming";

    static int N;
    static int B;
    static int D;

    static List<Integer> list = new ArrayList<>();
    // static Set<Code> visited = new HashSet<>();

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        B = Integer.parseInt(st.nextToken());
        D = Integer.parseInt(st.nextToken());

        list.add(0);
        int n = 0;
        while (list.size() < N) {
            n++;
            if (length(n) > B) {
                break;
            }
            if (distancesGood(n)) {
                // System.out.println("add " + n);
                list.add(n);
            }
        }
        int count = 0;
        for (int i = 0; i < list.size(); i++) {
            out.print(list.get(i));
            count++;
            if (count % 10 == 0 || i == list.size() - 1) {
                out.println();
                count = 0;
            } else {
                out.print(" ");
            }
        }
        // out.println();

        // System.out.println(distance(3, 2));
        // System.out.println(D);

        out.close();
    }

    public static boolean distancesGood(int n) {
        for (int k : list) {
            if (distance(k, n) < D) {
                return false;
            }
        }
        return true;
    }

    public static int length(int n) {
        int len = 0;
        while (n > 0) {
            n >>= 1;
            len++;
        }
        return len;
    }

    public static int distance(int m, int n) {
        int x = m ^ n;
        int d = 0;
        while (x > 0) {
            if ((x & 1) > 0) {
                d++;
            }
            x >>= 1;
        }
        return d;
    }

}
