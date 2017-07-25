/*
ID: libra_k1
LANG: JAVA
TASK: sort3
*/
import java.io.*;
import java.util.*;

class sort3 {

    private static String task = "sort3";

    static int N;
    static int[] nums;
    static int[] sorted_nums;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        nums = new int[N];
        sorted_nums = new int[N];
        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            nums[i] = Integer.parseInt(st.nextToken());
            sorted_nums[i] = nums[i];
        }
        Arrays.sort(sorted_nums);

        Set<State> visited = new HashSet<>();
        Deque<State> q = new ArrayDeque<>();
        State init = new State(nums);
        q.add(init);

        while (! q.isEmpty()) {
            State s = q.poll();
            visited.add(s);
            if (sorted(s)) {
                int nExchange = 0;
                while (s.parent != null) {
                    System.out.println(s);
                    nExchange++;
                    s = s.parent;
                }
                out.println(nExchange);
                break;
            }
            for (int i = 0; i < N-1; i++) {
                for (int j = i+1; j < N; j++) {
                    if (s.nums[i] != s.nums[j] && s.nums[i] != sorted_nums[i] && s.nums[j] != sorted_nums[j]) {
                        State ns = new State(s.nums, i, j);
                        ns.parent = s;
                        if (! visited.contains(ns)) {
                            q.add(ns);
                        }
                    }
                }
            }
        }

        out.close();
    }

    static boolean sorted(State s) {
        for (int i = 1; i < s.nums.length; i++) {
            if (s.nums[i-1] > s.nums[i]) {
                return false;
            }
        }
        return true;
    }

    static class State {
        public int[] nums;
        public State parent = null;

        public State(int[] nums) {
            this.nums = new int[nums.length];
            System.arraycopy(nums, 0, this.nums, 0, nums.length);
        }

        public State(int[] nums, int i, int j) {
            this.nums = new int[nums.length];
            System.arraycopy(nums, 0, this.nums, 0, nums.length);
            int tmp = this.nums[i];
            this.nums[i] = this.nums[j];
            this.nums[j] = tmp;
        }

        public int hashCode() {
            StringBuilder sb = new StringBuilder();
            for (int i : nums) {
                sb.append(i);
            }
            return sb.toString().hashCode();
        }

        public boolean equals(Object o) {
            if (! (o instanceof State)) {
                return false;
            }
            State s = (State) o;
            if (this.nums.length != s.nums.length) {
                return false;
            }
            for (int i = 0; i < nums.length; i++) {
                if (this.nums[i] != s.nums[i]) {
                    return false;
                }
            }
            return true;
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int n : this.nums) {
                sb.append(n);
                sb.append(' ');
            }
            return sb.toString();
        }
    }
}
