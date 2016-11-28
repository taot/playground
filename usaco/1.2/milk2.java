/*
ID: libra_k1
LANG: JAVA
TASK: milk2
*/
import java.io.*;
import java.util.*;

class milk2 {

    private static String task = "milk2";

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        int n = Integer.parseInt(st.nextToken());

        Range[] ranges = new Range[n];
        for (int i = 0; i < n; i++) {
            st = new StringTokenizer(f.readLine());
            int start = Integer.parseInt(st.nextToken());
            int end = Integer.parseInt(st.nextToken());
            ranges[i] = new Range(start, end);
        }

        Arrays.sort(ranges, new Comparator<Range>() {
            public int compare(Range r1, Range r2) {
                return r1.start - r2.start;
            }
        });

        int maxIdle = 0;
        // int idleStart = 0;
        // int idleEnd = ranges[0].start;
        int maxBusy = ranges[0].end - ranges[0].start;
        int busyStart = ranges[0].start;
        int busyEnd = ranges[0].end;

        for (int i = 1; i < n; i++) {
            // busy
            if (ranges[i].start <= busyEnd) {
                busyEnd = max(busyEnd, ranges[i].end);
            } else {
                maxBusy = max(maxBusy, busyEnd - busyStart);
                maxIdle = max(maxIdle, ranges[i].start - busyEnd);

                busyStart = ranges[i].start;
                busyEnd = ranges[i].end;
            }

            // idle
            // if (ranges[i].start > idleEnd) {
            //
            // }
        }

        out.println(maxBusy + " " + maxIdle);
        out.close();
    }

    private static int max(int a, int b) {
        if (a > b) {
            return a;
        } else {
            return b;
        }
    }

    private static class Range {
        int start;
        int end;
        public Range(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }
}
