/*
ID: libra_k1
LANG: JAVA
TASK: ttwo
*/
import java.io.*;
import java.util.*;

class ttwo {

    private static String task = "ttwo";

    static final int NORTH = 0;
    static final int EAST = 1;
    static final int SOUTH = 2;
    static final int WEST = 3;

    static int N = 10;
    static int[][] map = new int[N][N];
    static Pos farmer;
    static Pos cow;
    static int nFree = 0;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));



        // int i1 = Integer.parseInt(st.nextToken());
        // int i2 = Integer.parseInt(st.nextToken());
        // out.println(i1+i2);

        for (int i = 0; i < N; i++) {
            // StringTokenizer st = new StringTokenizer(f.readLine());
            String line = f.readLine();
            for (int j = 0; j < N; j++) {
                char c = line.charAt(j);
                if (c == '*') {
                    map[i][j] = 1;
                } else if (c == 'F') {
                    farmer = new Pos(j, i, NORTH);
                } else if (c == 'C') {
                    cow = new Pos(j, i, NORTH);
                } else {
                    nFree++;
                }
            }
        }

        // printMap();

        int steps = simulate();
        out.println(steps);

        out.close();
    }

    static int simulate() {
        int steps = 0;
        boolean fVisited = false;
        boolean cVisited = false;

        // while (! fVisited || ! cVisited) {
        while (steps < 160000) {
            steps++;
            move(farmer);
            move(cow);

            if (farmer.visited()) {
                fVisited = true;
            }
            if (cow.visited()) {
                cVisited = true;
            }

            farmer.visit();
            cow.visit();

            // System.out.println("\n===== Step " + steps + " =====");
            // printMap();

            if (farmer.x == cow.x && farmer.y == cow.y) {
                break;
            }
        }

        return farmer.x == cow.x && farmer.y == cow.y ? steps : 0;
    }

    static void move(Pos p) {
        int nx = 0;
        int ny = 0;
        int nd = 0;
        if (p.d == NORTH) {
            nx = p.x;
            ny = p.y - 1;
            nd = EAST;
        } else if (p.d == EAST) {
            nx = p.x + 1;
            ny = p.y;
            nd = SOUTH;
        } else if (p.d == SOUTH) {
            nx = p.x;
            ny = p.y + 1;
            nd = WEST;
        } else if (p.d == WEST) {
            nx = p.x - 1;
            ny = p.y;
            nd = NORTH;
        }

        if (nx >= 0 && nx <= N-1 && ny >= 0 && ny <= N-1 && map[ny][nx] == 0) {
            p.x = nx;
            p.y = ny;
        } else {
            p.d = nd;
        }
    }

    static void printMap() {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                char c = '.';
                if (farmer.x == j && farmer.y == i) {
                    c = 'F';
                } if (cow.x == j && cow.y == i) {
                    if (c == 'F') {
                        c = 'G';
                    } else {
                        c = 'C';
                    }
                } else if (map[i][j] == 1) {
                    c = '*';
                }
                System.out.print(c);
            }
            System.out.println();
        }
    }

    static class Pos {

        public int x;
        public int y;
        public int d;
        public BitSet set;

        public Pos(int x, int y, int d) {
            this.x = x;
            this.y = y;
            this.d = d;
            this.set = new BitSet();
        }

        public boolean visited() {
            return set.get(y * 40 + x * 4 + d);
        }

        public void visit() {
            set.set(y * 40 + x * 4 + d);
        }

        public String toString() {
            return String.format("(%d,%d)", x, y);
        }
    }
}
