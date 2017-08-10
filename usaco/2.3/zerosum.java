/*
ID: libra_k1
LANG: JAVA
TASK: zerosum
*/
import java.io.*;
import java.util.*;

class zerosum {

    private static String task = "zerosum";

    static char[] OPS = {' ', '+', '-'};
    static int N;
    static List<String> lines = new ArrayList<>();

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());

        // int sum = calc("1-2+3+4-5+6-7");
        // System.out.println(sum);

        char[] ops = new char[N-1];
        permute(ops, 0);

        Collections.sort(lines);
        for (String s : lines) {
            out.println(s);
        }

        out.close();
    }

    static void permute(char[] ops, int n) {
        if (n >= N-1) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < N; i++) {
                if (i > 0) {
                    sb.append(ops[i-1]);
                }
                sb.append(i+1);
            }
            String s = sb.toString();
            int res = calc(s);
            if (res == 0) {
                lines.add(s);
                // System.out.println(sb.toString() + " = " + res);
            }

            return;
        }
        for (char c : OPS) {
            ops[n] = c;
            permute(ops, n+1);
        }
    }

    static int calc(String line) {
        line = line.replaceAll(" ", "");
        int sum = 0;
        char[] chars = line.toCharArray();
        int n = 0;
        char lastOp = '+';
        for (int i = 0; i < chars.length; i++) {
            char c = chars[i];
            if (c >= '0' && c <= '9') {
                n *= 10;
                n += (chars[i] - '0');
            } else if (c == '+' || c == '-') {
                if (lastOp == '+') {
                    sum += n;
                } else {
                    sum -= n;
                }
                n = 0;
                lastOp = c;
            }
        }
        if (lastOp == '+') {
            sum += n;
        } else {
            sum -= n;
        }
        return sum;
    }
}
