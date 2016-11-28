/*
ID: libra_k1
LANG: JAVA
TASK: milk
*/
import java.io.*;
import java.util.*;

class milk {

    private static String task = "milk";

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        int N = Integer.parseInt(st.nextToken());
        int M = Integer.parseInt(st.nextToken());

        List<Farmer> farmers = new ArrayList<>();

        for (int i = 0; i < M; i++) {
            st = new StringTokenizer(f.readLine());
            int P = Integer.parseInt(st.nextToken());
            int A = Integer.parseInt(st.nextToken());
            farmers.add(new Farmer(P, A));
        }

        Collections.sort(farmers, new Comparator<Farmer>() {
            public int compare(Farmer f1, Farmer f2) {
                return f1.price - f2.price;
            }
        });

        int total = 0;
        int remain = N;
        for (int i = 0; i < M; i++) {
            Farmer fm = farmers.get(i);
            if (remain >= fm.amount) {
                total += fm.amount * fm.price;
                remain -= fm.amount;
            } else {
                total += remain * fm.price;
                remain = 0;
            }
            if (remain == 0) {
                break;
            }
        }

        out.println(total);

        out.close();
    }

    private static class Farmer {
        public int price;
        public int amount;
        public Farmer(int price, int amount) {
            this.price = price;
            this.amount = amount;
        }
    }
}
