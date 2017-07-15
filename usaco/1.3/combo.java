/*
ID: libra_k1
LANG: JAVA
TASK: combo
*/
import java.io.*;
import java.util.*;

class combo {

    private static final int KEY_LEN = 3;

    private static final String task = "combo";

    private static int N;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        int[] farmerKeys = readKeys(f);
        int[] masterKeys = readKeys(f);
        // System.out.println(N);
        // for (int i = 0; i < farmerKeys.length; i++) {
        //     System.out.print(farmerKeys[i]);
        //     System.out.print(masterKeys[i]);
        // }
        // System.out.println();

        int x = calc(N, farmerKeys, masterKeys);
        out.println(x);

        out.close();
    }

    private static int calc(int N, int[] fks, int[] mks) {
        Set<Integer> bs = new HashSet<Integer>();
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                for (int k = -2; k <= 2; k++) {
                    int x = f(fks[0] + i) * 10000 + f(fks[1] + j) * 100 + f(fks[2] + k);
                    bs.add(x);
                    int y = f(mks[0] + i) * 10000 + f(mks[1] + j) * 100 + f(mks[2] + k);
                    bs.add(y);
                }
            }
        }
        return bs.size();
    }

    private static int f(int a) {
        a = a % N;
        if (a < 0) {
            a = a + N;
        }
        return a;
    }

    private static int[] readKeys(BufferedReader f) throws IOException {
        StringTokenizer st = new StringTokenizer(f.readLine());
        int[] keys = new int[KEY_LEN];
        for (int i = 0; i < KEY_LEN; i++) {
            keys[i] = Integer.parseInt(st.nextToken());
        }
        return keys;
    }
}
