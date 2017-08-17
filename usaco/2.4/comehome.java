/*
ID: libra_k1
LANG: JAVA
TASK: comehome
*/
import java.io.*;
import java.util.*;

class comehome {

    private static String task = "comehome";

    static int N = 52;
    static int M = 0;
    static int P;

    static int[][] graph = new int[N][N];
    static boolean[] cow = new boolean[N];
    static boolean[] nodes = new boolean[N];
    static int[][] dists = new int[N][N];

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                graph[i][j] = -1;
                dists[i][j] = -1;
            }
        }

        P = Integer.parseInt(st.nextToken());

        for (int i = 0; i < P; i++) {
            st = new StringTokenizer(f.readLine());
            char a = st.nextToken().charAt(0);
            char b = st.nextToken().charAt(0);
            int d = Integer.parseInt(st.nextToken());
            int m = getIndex(a);
            if (hasCow(a)) {
                cow[m] = true;
            }
            int n = getIndex(b);
            if (m == n) {
                continue;
            }
            if (hasCow(b)) {
                cow[n] = true;
            }
            nodes[m] = true;
            nodes[n] = true;
            if (graph[m][n] < 0 || graph[m][n] > d) {
                graph[m][n] = graph[n][m] = dists[m][n] = dists[n][m] = d;
            }
        }

        // printGraph(dists);
        // printCows();

        walk();
        // printGraph(dists);

        int minI = -1;
        int min = Integer.MAX_VALUE;
        for (int i = 26; i < N - 1; i++) {
            if (cow[i] && min > dists[i][N-1]) {
                min = dists[i][N-1];
                minI = i;
            }
        }
        System.out.println(minI);
        out.println(String.format("%c %d", (char) ('A' + minI - 26), min));

        out.close();
    }

    static void walk() {
        for (int k = 0; k < N; k++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    if (i == j || ! nodes[i] || ! nodes[j] || ! nodes[k]) {
                        continue;
                    }
                    if (dists[i][k] >= 0 && dists[k][j] >= 0 && ( dists[i][j] < 0 || dists[i][j] > dists[i][k] + dists[k][j])) {
                        dists[i][j] = dists[j][i] = dists[i][k] + dists[k][j];
                    }
                }
            }
        }
    }

    static void printCows() {
        for (int i = 0; i < N; i++) {
            if (! nodes[i]) {
                continue;
            }
            System.out.print(cow[i] ? 1 : 0);
            System.out.print(' ');
        }
        System.out.println();
    }

    static void printGraph(int[][] graph) {
        for (int i = 0; i < N; i++) {
            if (! nodes[i]) {
                continue;
            }
            System.out.print(getChar(i) + ": ");
            for (int j = 0; j < N; j++) {
                if (! nodes[j]) {
                    continue;
                }
                System.out.print(getChar(j) + ":" + graph[i][j] + " ");
            }
            System.out.println();
        }
    }


    static boolean hasCow(char c) {
        return (c >= 'A' && c <= 'Y');
    }

    static char getChar(int idx) {
        if (idx >= 26 && idx <= 52) {
            return (char) (idx - 26 + 'A');
        }
        return (char) (idx + 'a');
    }

    static int getIndex(char c) {
        if (c >= 'A' && c <= 'Z') {
            return c - 'A' + 26;
        }
        return c - 'a';
    }
}
