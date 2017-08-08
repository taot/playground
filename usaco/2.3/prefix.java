/*
ID: libra_k1
LANG: JAVA
TASK: prefix
*/
import java.io.*;
import java.util.*;

class prefix {

    private static String task = "prefix";

    static List<char[]> P = new ArrayList<>();

    static char[] S;

    static int[] M;

    public static void main (String [] args) throws IOException {

        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        readInput();

        for (int i = 0; i < S.length; i++) {
            boolean found = false;
            for (int j = 0; j < P.size(); j++) {
                char[] p = P.get(j);
                boolean match = true;
                for (int k = 0; k < p.length; k++) {
                    if (i - k < 0 || S[i-k] != p[p.length-k-1]) {
                        match = false;
                        break;
                    }
                }
                if (match && (i == p.length - 1 || i >= p.length && M[i-p.length] >= 0)) {
                    found = true;
                    M[i] = j;
                    break;
                }
                M[i] = -1;
            }
        }

        {
            int i = 0;
            for (i = M.length - 1; i >= 0; i--) {
                if (M[i] >= 0) {
                    out.println(i + 1);
                    break;
                }
            }
            if (i < 0) {
                out.println(0);
            }
        }


        // for (int i = 0; i < M.length; i++) {
        //     System.out.print(M[i] + " ");
        // }

        out.close();
    }

    static void readInput() throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        String line = f.readLine();
        while (! ".".equals(line.trim())) {
            String[] parts = line.trim().split(" ");
            for (String s : parts) {
                P.add(s.toCharArray());
            }
            line = f.readLine();
        }
        StringBuilder sb = new StringBuilder();
        line = f.readLine();
        while (line != null) {
            sb.append(line.trim());
            line = f.readLine();
        }
        S = sb.toString().toCharArray();
        M = new int[S.length];
    }
}
