/*
ID: libra_k1
LANG: JAVA
TASK: fracdec
*/
import java.io.*;
import java.util.*;

class fracdec {

    private static String task = "fracdec";
    static int N;
    static int D;

    static Map<Integer, Integer> remainders = new HashMap<>();
    static int[] decimals = new int[50000];
    static int repeat = -2;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        D = Integer.parseInt(st.nextToken());

        int a = N / D;
        int r = N % D;
        remainders.put(r, -1);

        int idx = 0;
        do {
            r *= 10;
            // count++;
            decimals[idx] = r / D;
            r = r % D;
            if (r == 0) {
                break;
            }
            if (remainders.get(r) != null) {
                repeat = remainders.get(r) + 1;
                break;
            }
            remainders.put(r, idx);
            idx++;
            // System.out.println("r = " + r);
        } while (true);

        int count = idx + 1;
        // System.out.println("a = " + a);
        // for (int i = 0; i < count; i++) {
        //     System.out.print(decimals[i]);
        // }
        // System.out.println();
        // System.out.println("repeat = " + repeat);

        StringBuilder sb = new StringBuilder();
        sb.append(a);
        sb.append(".");
        for (int i = 0; i < count; i++) {
            if (repeat >= -1 && repeat == i) {
                sb.append("(");
            }
            sb.append(decimals[i]);
        }
        if (repeat >= -1) {
            sb.append(")");
        }

        int start = 0;
        while (sb.length() - start > 0) {
            out.println(sb.substring(start, Math.min(start + 76, sb.length())));
            start += 76;
        }

        out.close();
    }
}
