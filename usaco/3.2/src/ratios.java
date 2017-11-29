/*
ID: libra_k1
LANG: JAVA
TASK: ratios
*/
import java.io.*;
import java.util.*;

class ratios {

    private static String task = "ratios";

    static int[] goals = new int[3];
    static int[][] mixtures = new int[3][3];

    static int[] minUnits = new int[3];
    static int minMultiple;
    static int minSum = Integer.MAX_VALUE;

    public static void main (String [] args) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));

        readData(reader);

        find();

        if (minSum == Integer.MAX_VALUE) {
            out.println("NONE");
            System.out.println("NONE");
        } else {
            for (int i = 0; i < 3; i++) {
                out.print(minUnits[i] + " ");
                System.out.print(minUnits[i] + " ");
            }
            out.println(minMultiple);
            System.out.println(minMultiple);
        }


        out.close();
    }

    static void find() {
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                for (int k = 0; k < 100; k++) {
                    int[] mixture = new int[3];
                    mixture[0] = mixtures[0][0] * i + mixtures[1][0] * j + mixtures[2][0] * k;
                    mixture[1] = mixtures[0][1] * i + mixtures[1][1] * j + mixtures[2][1] * k;
                    mixture[2] = mixtures[0][2] * i + mixtures[1][2] * j + mixtures[2][2] * k;
//                    for (int s = 0; s < 3; s++) {
//                        mixture[s] = mixtures[0][s] * i +
//                                mixtures[1][s] * j + mixtures[2][s] * k;
//                    }
                    int m = isMultiple(mixture);
                    if (m > 0 && (i+j+k) < minSum) {
                        minUnits[0] = i;
                        minUnits[1] = j;
                        minUnits[2] = k;
                        minMultiple = m;
                        minSum = i+j+k;
                    }
                }
            }
        }
    }

    static int isMultiple(int[] mixture) {
        if (mixture[0] % goals[0] != 0) {
            return -1;
        }
        int m = mixture[0] / goals[0];
        if (mixture[1] == m * goals[1] && mixture[2] == m * goals[2]) {
            return m;
        } else {
            return -1;
        }
    }

    static void readData(BufferedReader reader) throws IOException {
        StringTokenizer st = new StringTokenizer(reader.readLine());
        for (int i = 0; i < 3; i++) {
            goals[i] = Integer.parseInt(st.nextToken());
        }
        for (int i = 0; i < 3; i++) {
            st = new StringTokenizer(reader.readLine());
            for (int j = 0; j < 3; j++) {
                mixtures[i][j] = Integer.parseInt(st.nextToken());
            }
        }
    }
}
