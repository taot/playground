/*
ID: libra_k1
LANG: JAVA
TASK: game1
*/
import java.io.*;
import java.util.*;

class game1 {

    private static String task = "game1";

    static int N;
    static int board[];

    static int MIN[][];
    static int MAX[][];
    static int sum = 0;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        board = new int[N];
        MIN = new int[N+1][N+1];
        MAX = new int[N+1][N+1];
        st = new StringTokenizer(f.readLine());
        for (int i = 0; i < N; i++) {
            if (! st.hasMoreTokens()) {
                st = new StringTokenizer(f.readLine());
            }
            board[i] = Integer.parseInt(st.nextToken());
            sum += board[i];
        }

//        for (int i = 0; i < N; i++) {
//            System.out.println(board[i]);
//        }

        for (int i = 0; i < N; i++) {
            MIN[i][i] = MAX[i][i] = 0;
        }

        for (int s = 1; s <= N; s++) {
            for (int i = 0; i < N - s + 1; i++) {
                if ((N + s) % 2 == 0) {
                    MAX[i][s] = Math.max(MIN[i+1][s-1] + board[i], MIN[i][s-1] + board[i+s-1]);
                } else {
                    MIN[i][s] = Math.min(MAX[i+1][s-1] - board[i], MAX[i][s-1] - board[i+s-1]);
                }
            }
        }

//        System.out.println("\nMAX");
//        print(MAX);
//        System.out.println("\nMIN");
//        print(MIN);


        int d = MAX[0][N];
        int n1 = (sum + d) / 2;
        int n2 = (sum - d) / 2;
        System.out.println(n1 + " " + n2);
        out.println(n1 + " " + n2);

        out.close();
    }

    static void print(int[][] m) {
        for (int i = 0; i < N+1; i++) {
            for (int j = 0; j < N+1; j++) {
                System.out.print(m[i][j] + " ");
            }
            System.out.println();
        }
    }
}
