/*
ID: libra_k1
LANG: JAVA
TASK: shuttle
*/
import java.io.*;
import java.util.*;

class shuttle {

    private static String task = "shuttle";

    static PrintWriter out;

    static int N;
    static State GOAL;

    static Set<State> visited = new HashSet<>();
    static State result;
    static int resultLength;

    public static void main (String [] args) throws IOException {
        long start = System.currentTimeMillis();
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        GOAL = getGoalState();

//        bfs();

        for (int limit = 20; limit < 200; limit+= 20) {
            visited.clear();
            result = null;
            resultLength = Integer.MAX_VALUE;
            State s = getInitState();
            dfs(s, limit, 0);
            if (result != null) {
                printResult(result);
                break;
            }
        }

        out.close();

        System.out.println("Duration: " + (System.currentTimeMillis() - start) + " ms");
    }

    static void dfs(State s, int limit, int depth) {
        visited.add(s);
        if (s.isGoal()) {
            if (s.length <= resultLength && (result == null || compare(s, result) < 0)) {
                result = s;
                resultLength = s.length;
            }
            return;
        }

        if (s.length >= resultLength) {
            return;
        }
        if (depth >= limit) {
            return;
        }

        List<State> moves = s.getMoves();

        Collections.sort(moves, new Comparator<State>() {
            @Override
            public int compare(State s1, State s2) {
                return (s1.dist() - s2.dist());
            }
        });

        for (State m : moves) {
            if (visited.contains(m)) {
                continue;
            }
            dfs(m, limit, depth + 1);
        }
    }

    static void bfs() {
        State s = getInitState();
        Deque<State> q = new ArrayDeque<>();
        Set<State> visited = new HashSet<>();
        q.addFirst(s);
        visited.add(s);

        while ((s = q.pollLast()) != null) {
            List<State> moves = s.getMoves();
            for (State m : moves) {
                if (m.isGoal()) {
                    printResult(m);
                    return;
                }
                if (! visited.contains(m)) {
                    visited.add(m);
                    q.addFirst(m);
                }
            }
        }
    }

    static List<Integer> getResult(State s) {
        if (s == null) {
            return null;
        }

        List<Integer> list = new ArrayList<>();
        while (s.prev != null) {
            list.add(s.move + 1);
            s = s.prev;
        }
        Collections.reverse(list);
        return list;
    }

    static void printResult(State s) {
        if (s == null) {
            return;
        }

        List<Integer> list = new ArrayList<>();
        while (s.prev != null) {
            list.add(s.move + 1);
            s = s.prev;
        }
        Collections.reverse(list);
        for (int i = 0; i < list.size(); i++) {
            if (i > 0 && i % 20 == 0) {
                System.out.println();
                out.println();
            }
            if (i % 20 != 0) {
                System.out.print(' ');
                out.print(' ');
            }
            System.out.print(list.get(i));
            out.print(list.get(i));
        }
        System.out.println();
        out.println();

        System.out.println(list.size());
    }

    static State getInitState() {
        char[] puzzle = new char[N * 2 + 1];
        for (int i = 0; i < N; i++) {
            puzzle[i] = 'w';
            puzzle[N + 1 + i] = 'b';
        }
        puzzle[N] = ' ';
        return new State(puzzle, null, -1);
    }

    static State getGoalState() {
        char[] puzzle = new char[N * 2 + 1];
        for (int i = 0; i < N; i++) {
            puzzle[i] = 'b';
            puzzle[N + 1 + i] = 'w';
        }
        puzzle[N] = ' ';
        return new State(puzzle, null, -1);
    }

    static int compare(State s1, State s2) {
        List<Integer> r1 = getResult(s1);
        List<Integer> r2 = getResult(s2);
        if (r1.size() < r2.size()) {
            return -1;
        } else if (r1.size() > r2.size()) {
            return 1;
        }

        for (int i = 0; i < r1.size(); i++) {
            int n1 = r1.get(i);
            int n2  = r2.get(i);
            if (n1 < n2) {
                return -1;
            } else if (n1 > n2) {
                return 1;
            }
        }
        return 0;
    }

    static class State {
        public int move;
        public char[] puzzle;
        public State prev;
        public int length = 0;

        public State(char[] p, State prev, int move) {
            puzzle = new char[p.length];
            System.arraycopy(p, 0, puzzle, 0, p.length);
            this.move = move;
            this.prev = prev;
            if (prev != null) {
                this.length = prev.length + 1;
            }
        }

        public List<State> getMoves() {
            int x;
            for (x = 0; x < puzzle.length; x++) {
                if (puzzle[x] == ' ') {
                    break;
                }
            }
            State s;
            List<State> moves = new ArrayList<>();

            if (x > 1 && puzzle[x-2] != puzzle[x-1]) {
                s = new State(puzzle, this, x-2);
                s.puzzle[x] = s.puzzle[x-2];
                s.puzzle[x-2] = ' ';
                moves.add(s);
            }

            if (x > 0) {
                s = new State(puzzle, this, x-1);
                s.puzzle[x] = s.puzzle[x-1];
                s.puzzle[x-1] = ' ';
                moves.add(s);
            }

            if (x < puzzle.length - 1) {
                s = new State(puzzle, this, x+1);
                s.puzzle[x] = s.puzzle[x+1];
                s.puzzle[x+1] = ' ';
                moves.add(s);
            }


            if (x < puzzle.length - 2 && puzzle[x+2] != puzzle[x+1]) {
                s = new State(puzzle, this, x+2);
                s.puzzle[x] = s.puzzle[x+2];
                s.puzzle[x+2] = ' ';
                moves.add(s);
            }

            return moves;
        }

        public int dist() {
            int d = 0;
            for (int i = 0; i < puzzle.length; i++) {
                if (puzzle[i] != GOAL.puzzle[i]) {
                    d += 1;
                }
            }
            return d;
        }

        public boolean isGoal() {
            for (int i = 0; i < puzzle.length; i++) {
                if (puzzle[i] != GOAL.puzzle[i]) {
                    return false;
                }
            }
            return true;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            State state = (State) o;
            return length == state.length &&
                    Arrays.equals(puzzle, state.puzzle);
        }

        @Override
        public int hashCode() {

            int result = Objects.hash(length);
            result = 31 * result + Arrays.hashCode(puzzle);
            return result;
        }

        @Override
        public String toString() {
            return "State{" +
                    "puzzle=" + Arrays.toString(puzzle) +
                    '}';
        }
    }
}
