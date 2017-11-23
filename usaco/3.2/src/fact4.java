/*
ID: libra_k1
LANG: JAVA
TASK: fact4
*/
import java.io.*;
import java.util.*;

class fact4 {

    private static String task = "fact4";

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        int N = Integer.parseInt(st.nextToken());

        int d = facLastDigit(N);

        System.out.println(d);
        out.println(d);
        out.close();
    }

    private static int facLastDigit(int n) {
        int count2 = 0;
        int fac = 1;
        for (int i = 2; i <= n; i++) {
            int x = i;
            while (x % 2 == 0) {
                x /= 2;
                count2++;
            }
            while (x % 5 == 0) {
                x /= 5;
                count2--;
            }
            fac = (fac * x) % 10;
        }
        while (count2 > 0) {
            fac = (fac * 2) % 10;
            count2--;
        }
        while (count2 < 0) {
            fac = (fac * 5) % 10;
            count2++;
        }
        return fac;
    }
}