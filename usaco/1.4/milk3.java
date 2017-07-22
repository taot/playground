/*
ID: libra_k1
LANG: JAVA
TASK: milk3
*/
import java.io.*;
import java.util.*;

class milk3 {

    private static String task = "milk3";

    private static int[] cap = new int[3];


    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        cap[0] = Integer.parseInt(st.nextToken());
        cap[1] = Integer.parseInt(st.nextToken());
        cap[2] = Integer.parseInt(st.nextToken());

        State init =  new State(new int[] { 0, 0, cap[2] });
        Set<Integer> results = walk(init);

        boolean first = true;
        for (Integer n : results) {
            if (first) {
                first = false;
            } else {
                out.print(" ");
            }
            out.print(n);
        }
        out.println();

        out.close();
    }

    private static Set<Integer> walk(State init) {
        Deque<State> q = new ArrayDeque<State>();
        Set<State> set = new HashSet<State>();
        BitSet bitSet = new BitSet();
        q.add(init);
        set.add(init);

        Set<Integer> results = new TreeSet<Integer>();

        while (! q.isEmpty()) {
            State s = q.poll();
            if (s.b[0] == 0) {
                results.add(s.b[2]);
            }
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    State n = s.transfer(i, j);
                    if (n != null) {
                        int f = n.feature();
                        if (! set.contains(f)) {
                        // if (! bitSet.get(f)) {
                            q.add(n);
                            // bitSet.set(f);
                            set.add(n);
                        }
                    }
                }
            }
        }

        return results;
    }

    public static class State {

        public int[] b = new int[3];

        public State(int[] b) {
            for (int i = 0; i < b.length; i++) {
                this.b[i] = b[i];
            }
        }

        public State transfer(int from, int to) {
            if (from == to) {
                return null;
            }
            if (b[from] == 0 || b[to] == cap[to]) {
                return null;
            }
            State s = new State(this.b);
            int c = cap[to] - s.b[to];
            if (s.b[from] > c) {
                s.b[from] -= c;
                s.b[to] = cap[to];
            } else {
                s.b[to] = s.b[to] + s.b[from];
                s.b[from] = 0;
            }
            return s;
        }

        public boolean equals(Object o) {
            if (! (o instanceof State)) {
                return false;
            }
            State s = (State) o;
            for (int i = 0; i < 3; i++) {
                if (this.b[i] != s.b[i]) {
                    return false;
                }
            }
            return true;
        }

        public int hashCode() {
            StringBuilder sb = new StringBuilder();
            int sum = 0;
            for (int i = 0; i < 3; i++) {
                sum += this.b[i];
            }
            // int s = sb.toString().hashCode();
            // System.out.println("state: " + this);
            // System.out.println("hashCode: " + s);
            return sum;
        }

        public int feature() {
            int n = 0;
            for (int i = 0; i < 3; i++) {
                n *= 40;
                n += this.b[i];
            }
            return n;
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 3; i++) {
                sb.append(this.b[i]);
                sb.append(",");
            }
            return sb.toString();
        }
    }
}
