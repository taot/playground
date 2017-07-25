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

    static int nRooms;
    static List<Integer> roomSizes = new ArrayList<>();
    static Node nodeOfWallToRemove;
    static int roomSizeAfterRemoval;
    static char wallDirection;

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
        findComponents();

        // printGraph(true);

        // num of rooms
        out.println(nRooms);

        // largest room
        int maxSize = -1;
        for (Integer n : roomSizes) {
            if (n > maxSize) {
                maxSize = n;
            }
        }
        out.println(maxSize);

        // remove wall
        removeWall();
        out.println(roomSizeAfterRemoval);
        out.println(String.format("%d %d %c", nodeOfWallToRemove.row + 1, nodeOfWallToRemove.col + 1, wallDirection));

        out.close();
    }

    static void removeWall() {
        nodeOfWallToRemove = null;
        roomSizeAfterRemoval = -1;
        wallDirection = ' ';

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                Node n = getNode(i, j);
                if (i > 0 && n.neighbors[NORTH_IDX] == null) {
                    Node north = getNode(i-1, j);
                    if (n.component != north.component) {
                        int size = roomSizes.get(n.component) + roomSizes.get(north.component);
                        if (size > roomSizeAfterRemoval || (size == roomSizeAfterRemoval && shouldReplaceWall(nodeOfWallToRemove, n, 'N'))) {
                            roomSizeAfterRemoval = size;
                            nodeOfWallToRemove = n;
                            wallDirection = 'N';
                        }
                    }
                }
                if (j < M-1 && n.neighbors[EAST_IDX] == null) {
                    Node east = getNode(i, j+1);
                    if (n.component != east.component) {
                        int size = roomSizes.get(n.component) + roomSizes.get(east.component);
                        if (size > roomSizeAfterRemoval || (size == roomSizeAfterRemoval && shouldReplaceWall(nodeOfWallToRemove, n, 'E'))) {
                            roomSizeAfterRemoval = size;
                            nodeOfWallToRemove = n;
                            wallDirection = 'E';
                        }
                    }
                }
            }
        }
    }

    static boolean shouldReplaceWall(Node orig, Node toReplace, char d2) {
        if (orig.col < toReplace.col) {
            return false;
        }
        if (orig.col > toReplace.col) {
            return true;
        }
        if (orig.row > toReplace.row) {
            return false;
        }
        if (orig.row < toReplace.row) {
            return true;
        }
        if (d2 == 'N') {
            return true;
        }
        return false;
    }

    static void findComponents() {
        // clear
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                Node n = getNode(i, j);
                n.visited = false;
                n.component = -1;
            }
        }
        roomSizes.clear();

        // start searching
        Node notVisited = findNotVisited();
        int component = 0;
        Deque<Node> queue = new ArrayDeque<>();
        while (notVisited != null) {
            int roomSize = 0;
            queue.add(notVisited);
            notVisited.visited = true;
            roomSize++;
            while (! queue.isEmpty()) {
                Node n = queue.poll();

                n.component = component;
                for (int i = 0; i < 4; i++) {
                    Node m = n.neighbors[i];
                    if (m != null && ! m.visited) {
                        queue.add(m);
                        m.visited = true;
                        roomSize++;
                    }
                }
            }
            roomSizes.add(roomSize);

            notVisited = findNotVisited();
            component++;
        }

        nRooms = component;
    }

    static Node findNotVisited() {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                Node n = getNode(i, j);
                if (! n.visited) {
                    return n;
                }
            }
        }
        return null;
    }

    static void printGraph(boolean printComponent) {
        for (int i = 0; i < N; i++) {
            StringBuilder l1 = new StringBuilder();
            StringBuilder l2 = new StringBuilder();
            for (int j = 0; j < M; j++) {
                Node n = getNode(i, j);
                if (printComponent) {
                    l1.append(n.component);
                } else {
                    l1.append("#");
                }
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
        public boolean visited = false;
        public int component;

        public Node(int row, int col) {
            this.row = row;
            this.col = col;
        }
    }
}
