/*
ID: libra_k1
LANG: JAVA
TASK: gift1
*/
import java.io.*;
import java.util.*;

class gift1 {

    private static String task = "gift1";

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));

        StringTokenizer st = new StringTokenizer(f.readLine());
        int np = Integer.parseInt(st.nextToken());

        Map<String, Integer> map = new LinkedHashMap<String, Integer>();
        for (int i = 0; i < np; i++) {
            String s = f.readLine();
            map.put(s, 0);
        }

        for (int i = 0; i < np; i++) {
            String giver = f.readLine();
            // System.out.println("giver: " + giver);
            st = new StringTokenizer(f.readLine());
            int money = Integer.parseInt(st.nextToken());
            int n = Integer.parseInt(st.nextToken());
            // System.out.println("money:" + money + " n: " + n);
            // if (money == 0) {
            //     continue;
            // }
            // if (n == 0) {
            //     int amt = map.get(giver);
            //     map.put(giver, amt + money);
            //     continue;
            // }
            int giveAmt = 0;
            int rem = 0;
            if (n != 0) {
                giveAmt = money / n;
                rem = money % n;
            }
            for (int j = 0; j < n; j++) {
                String receiver = f.readLine();
                int amt = map.get(receiver);
                map.put(receiver, amt + giveAmt);
            }
            int amt = map.get(giver);
            map.put(giver, amt - money + rem);
        }

        for (String name : map.keySet()) {
            out.println(name + ' ' + map.get(name));
        }

        // out.println(i1+i2);
        out.close();
    }
}
