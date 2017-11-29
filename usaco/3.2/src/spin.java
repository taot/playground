/*
ID: libra_k1
LANG: JAVA
TASK: spin
*/
import java.io.*;
import java.util.*;

class spin {

    private static String task = "spin";
    static int N = 5;
    static Wheel[] wheels = new Wheel[N];

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        read(f);

        boolean found = false;
        int n = 0;
        for (n = 0; n < 360; n++) {
            if (good()) {
                out.println(n);
                System.out.println(n);
                found = true;
                break;
            }
            simulate();
        }
        if (! found && good()) {
            out.println(n);
            System.out.println(n);
            found = true;
        }
        if (! found) {
            out.println("none");
            System.out.println("none");
        }

        out.close();
    }

    static void simulate() {
        for (int i = 0; i < N; i++) {
            Wheel w = wheels[i];
            for (int j = 0; j < w.nWedges; j++) {
                w.wedgeStarts[j] = normalize(w.wedgeStarts[j] + w.speed);
                w.wedgeEnds[j] = normalize(w.wedgeEnds[j] + w.speed);
            }
        }
    }

    static int normalize(int d) {
        while (d > 360) {
            d -= 360;
        }
        return d;
    }

    static boolean good() {
        for (int d = 0; d < 360; d++) {
            boolean flag = true;
            for (int i = 0; i < wheels.length; i++) {
                Wheel w = wheels[i];
                if (! through(d, w)) {
                    flag = false;
                    continue;
                }
            }
            if (flag) {
                return true;
            }
        }
        return false;
    }

    static boolean through(int degree, Wheel w) {
        for (int i = 0; i < w.nWedges; i++) {
            if (w.wedgeStarts[i] <= w.wedgeEnds[i]) {
                if (degree >= w.wedgeStarts[i] && degree <= w.wedgeEnds[i]) {
                    return true;
                }
            } else {
                if (degree >= w.wedgeStarts[i] || degree <= w.wedgeEnds[i]) {
                    return true;
                }
            }
        }
        return false;
    }

    static void read(BufferedReader reader) throws IOException {
        for (int i = 0; i < N; i++) {
            StringTokenizer st = new StringTokenizer(reader.readLine());
            Wheel w = new Wheel();
            w.speed = Integer.parseInt(st.nextToken());
            w.nWedges = Integer.parseInt(st.nextToken());
            w.wedgeStarts = new int[w.nWedges];
            w.wedgeEnds = new int[w.nWedges];
            for (int j = 0; j < w.nWedges; j++) {
                w.wedgeStarts[j] = Integer.parseInt(st.nextToken());
                w.wedgeEnds[j] = w.wedgeStarts[j] + Integer.parseInt(st.nextToken());
            }
            wheels[i] = w;
        }

    }

    static class Wheel {
        public int speed;
        public int nWedges;
        public int[] wedgeStarts;
        public int[] wedgeEnds;
        public int alignMark;
    }
}
