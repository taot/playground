/*
ID: libra_k1
LANG: JAVA
TASK: beads
*/
import java.io.*;
import java.util.*;

class beads {

    private static String task = "beads";

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());


        int n = Integer.parseInt(st.nextToken());
        String line = f.readLine();
        char[] beads = line.toCharArray();

        if (isSameColor(beads)) {
            out.println(n);
        } else {
            int max = calc(beads);
            out.println(max);
        }

        // System.out.println(beads.length);

        out.close();
    }

    private static class Item {
        int count;
        char color;
    }

    // private static int calc2(char[] beads) {
    //     char[] joint = join(beads);
    //     List<Item> grouped = new ArrayList<Item>();
    //     char c = beads[0];
    //     int count = 1;
    //     for (int i = 1; i < joint.length; i++) {
    //         if (joint[i] == c) {
    //             count++;
    //         } else {
    //             Item it = new Item();
    //             it.count = count;
    //             it.color = c;
    //             grouped.add(it);
    //         }
    //     }
    //     Item it = new Item();
    //     it.count = count;
    //     it.color = c;
    //     grouped.add(it);
    //
    //     for (int i = 0; i < grouped.size() - 1; i++) {
    //
    //         for (int j = 0; j < grouped.size(); j++) {
    //
    //         }
    //     }
    //     return 0;
    // }

    private static char[] join(char[] beads) {
        char[] joint = new char[beads.length * 2];
        System.arraycopy(beads, 0, joint, 0, beads.length);
        System.arraycopy(beads, 0, joint, beads.length, beads.length);
        return joint;
    }

    private static int calc(char[] beads) {
        char[] joint = join(beads);
        // System.out.println(new String(joint));
        int max = 0;
        int n = beads.length;
        // int lastSegCount = 0;

        for (int i = 0; i < n * 2 - 1; i++) {
            int count = 1;
            char c = joint[i];
            boolean flipped = false;
            int limit = 2 * n;
            if (limit > n + i) {
                limit = n + i;
            }
            for (int j = i + 1; j < limit; j++) {
                if (c == 'w') {
                    count++;
                    c = joint[j];
                    continue;
                }
                if (joint[j] == c || joint[j] == 'w') {
                    count++;
                } else if (! flipped) {
                    flipped = true;
                    c = joint[j];
                    count++;
                } else {
                    break;
                }
            }
            if (count > max) {
                max = count;
            }
        }
        return max;
    }

    private static boolean isSameColor(char[] beads) {
        int nWhite = 0;
        int nBlue = 0;
        int nRed = 0;
        for (int i = 0; i < beads.length; i++) {
            char c = beads[i];
            if (c == 'w') {
                nWhite++;
            } else if (c == 'r') {
                nRed++;
            } else {
                nBlue++;
            }
        }
        return nBlue == 0 || nRed == 0;
    }
}
