/*
ID: libra_k1
LANG: JAVA
TASK: holstein
*/
import java.io.*;
import java.util.*;

class holstein {

    private static String task = "holstein";

    static PrintWriter out;

    static int V;
    static int[] requires;
    static int G;
    static int[][] feeds;
    static BitSet bitSet = new BitSet();

    public static void main (String [] args) throws IOException {
        // read inputs
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        V = Integer.parseInt(st.nextToken());
        requires = new int[V];
        st = new StringTokenizer(f.readLine());
        for (int i = 0; i < V; i++) {
            requires[i] = Integer.parseInt(st.nextToken());
        }
        st = new StringTokenizer(f.readLine());
        G = Integer.parseInt(st.nextToken());
        feeds = new int[G][V];
        for (int i = 0; i < G; i++) {
            st = new StringTokenizer(f.readLine());
            for (int j = 0; j < V; j++) {
                feeds[i][j] = Integer.parseInt(st.nextToken());
            }
        }

        Deque<Integer> q = new ArrayDeque<>();
        int init = 0;
        q.add(init);
        bitSet.set(init);
        while (! q.isEmpty()) {
            int s = q.poll();
            if (satisfied(s)) {
                printResult(s);
                break;
            }

            for (int i = 0; i < G; i++) {
                int m = 1 << i;
                if ((s & m) == 0) {
                    int s1 = s | m;
                    if (! bitSet.get(s1)) {
                        q.add(s1);
                        bitSet.set(s1);
                    }
                }
            }
        }

        out.close();
    }

    static void printResult(int s) {
        int count = 0;
        for (int i = 0; i < G; i++) {
            int m = 1 << i;
            if ((m & s) != 0) {
                count++;
            }
        }
        out.print(count + " ");
        boolean first = true;
        for (int i = 0; i < G; i++) {
            int m = 1 << i;
            if ((m & s) != 0) {
                if (first) {
                    first = false;
                } else {
                    out.print(" ");
                }
                out.print(i+1);
            }
        }
        out.println();
    }

    static boolean satisfied(int s) {
        int[] vitamins = new int[V];
        for (int i = 0; i < G; i++) {
            int m = 1 << i;
            if ((m & s) != 0) {
                for (int j = 0; j < V; j++) {
                    vitamins[j] += feeds[i][j];
                }
            }
        }
        for (int i = 0; i < V; i++) {
            if (vitamins[i] < requires[i]) {
                return false;
            }
        }
        return true;
    }
}
