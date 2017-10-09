/*
ID: libra_k1
LANG: JAVA
TASK: inflate
*/
import java.io.*;
import java.util.*;

class inflate {

    private static String task = "inflate";

    static int M;
    static int N;
    static int[] points;
    static int[] minutes;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        M = Integer.parseInt(st.nextToken());
        N = Integer.parseInt(st.nextToken());
        points = new int[N];
        minutes = new int[N];


        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            points[i] = Integer.parseInt(st.nextToken());
            minutes[i] = Integer.parseInt(st.nextToken());
        }

        dp();

        for (int i = M; i >= 0; i--) {
            if (scores[i] > 0) {
                System.out.println(scores[i]);
                out.println(scores[i]);
                break;
            }
        }

        out.close();
    }

    static int scores[];

    static void dp() {
        scores = new int[M+1];
        scores[0] = 0;
        for (int i = 1; i < M+1; i++) {
            scores[i] = 0;
        }
        for (int i = 0; i <= M; i++) {
            int max = 0;
            for (int j = 0; j < N; j++) {
                int min2 = i - minutes[j];
                if (min2 >= 0 && scores[min2] + points[j] > max) {
                    max = scores[min2] + points[j];
                }
            }
            scores[i] = max;
        }
    }
}
