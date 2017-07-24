/*
ID: libra_k1
LANG: JAVA
TASK: castle
*/
import java.io.*;
import java.util.*;

class castle {

    private static String task = "castle";

    static int M;
    static int N;

    static int[][] wall_arr;
    static List<List<Node>> graph;

    static boolean[][] graph2;

    static final int WEST_MASK = 1;
    static final int NORTH_MASK = 2;
    static final int EAST_MASK = 4;
    static final int SOUTH_MASK = 8;

    static final int WEST_IDX = 0;
    static final int NORTH_IDX = 1;
    static final int EAST_IDX = 2;
    static final int SOUTH_IDX = 3;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        M = Integer.parseInt(st.nextToken());
        N = Integer.parseInt(st.nextToken());
        wall_arr = new int[N][M];
        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            for (int j = 0; j < M; j++) {
                wall_arr[i][j] = Integer.parseInt(st.nextToken());
            }
        }

        createGraph();

        printGraph();

        out.close();
    }

    static void printGraph() {
        for (int i = 0; i < N; i++) {
            StringBuilder l1 = new StringBuilder();
            StringBuilder l2 = new StringBuilder();
            for (int j = 0; j < M; j++) {
                Node n = getNode(i, j);
                l1.append("#");
                if (n.neighbors[EAST_IDX] != null) {
                    l1.append("-");
                } else {
                    l1.append(" ");
                }
                if (n.neighbors[SOUTH_IDX] != null) {
                    l2.append("| ");
                } else {
                    l2.append("  ");
                }
            }
            System.out.println(l1.toString());
            System.out.println(l2.toString());
        }
    }

    static void createGraph() {
        graph = new ArrayList<List<Node>>();
        for (int i = 0; i < N; i++) {
            List<Node> list = new ArrayList<Node>();
            graph.add(list);
            for (int j = 0; j < M; j++) {
                list.add(new Node(i, j));
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                int x = wall_arr[i][j];
                Node n = getNode(i, j);
                if (j != 0) {
                    if ((x & WEST_MASK) == 0) {
                        Node m = getNode(i, j-1);
                        n.neighbors[WEST_IDX] = m;
                        m.neighbors[EAST_IDX] = n;
                    }
                }
                if (j != M-1) {
                    if ((x & EAST_MASK) == 0) {
                        Node m = getNode(i, j+1);
                        n.neighbors[EAST_IDX] = m;
                        m.neighbors[WEST_IDX] = n;
                    }
                }
                if (i != 0) {
                    if ((x & NORTH_MASK) == 0) {
                        Node m = getNode(i-1, j);
                        n.neighbors[NORTH_IDX] = m;
                        m.neighbors[SOUTH_IDX] = n;
                    }
                }
                if (i != N-1) {
                    if ((x & SOUTH_MASK) == 0) {
                        Node m = getNode(i+1, j);
                        n.neighbors[SOUTH_IDX] = m;
                        m.neighbors[NORTH_IDX] = n;
                    }
                }
            }
        }
    }

    static Node getNode(int row, int col) {
        return graph.get(row).get(col);
    }

    static class Node {
        public int row;
        public int col;
        public Node[] neighbors = new Node[4];
        public boolean visisted = false;

        public Node(int row, int col) {
            this.row = row;
            this.col = col;
        }
    }
}
