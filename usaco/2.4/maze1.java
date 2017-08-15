/*
ID: libra_k1
LANG: JAVA
TASK: maze1
*/
import java.io.*;
import java.util.*;

class maze1 {

    private static String task = "maze1";

    static int W;
    static int H;
    static char[][] maze;
    // static List[][] graph;
    // static boolean[][] exits;
    static List<Point> exits = new ArrayList<>();
    static boolean[][] visited;
    static int[][] minDists;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());

        W = Integer.parseInt(st.nextToken());
        H = Integer.parseInt(st.nextToken());

        maze = new char[2*H+1][2*W+1];
        for (int i = 0; i < 2*H+1; i++) {
            String line = f.readLine();
            for (int j = 0; j < 2*W+1; j++) {
                char c = ' ';
                if (j < line.length()) {
                    c = line.charAt(j);
                }
                maze[i][j] = c;
            }
        }
        findExits();

        printMaze();
        System.out.println();
        printGraph();
        System.out.println();

        // printExits();

        minDists = new int[H][W];
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                minDists[i][j] = Integer.MAX_VALUE;
            }
        }

        // int minDist = Integer.MAX_VALUE;
        for (Point s : exits) {
            bfs(s);
        }
        // out.println(minDist + 1);
        int dist = 0;
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                if (dist < minDists[i][j]) {
                    dist = minDists[i][j];
                }
            }
        }
        out.println(dist + 1);

        out.close();
    }

    static int bfs(Point start) {
        // System.out.println("bfs");
        visited = new boolean[H][W];
        Deque<Point> q = new ArrayDeque<>();
        q.addLast(start);
        visited[start.y][start.x] = true;
        if (minDists[start.y][start.x] > start.dist) {
            minDists[start.y][start.x] = start.dist;
        }
        Point p;
        int maxDist = 0;
        while ((p = q.pollFirst()) != null) {
            // System.out.println(p);
            if (p.dist > maxDist) {
                maxDist = p.dist;
            }
            if (minDists[p.y][p.x] > p.dist) {
                minDists[p.y][p.x] = p.dist;
            }
            List<Point> ns = neighbors(p);
            // System.out.println("neighbors: " + ns);
            for (Point n : ns) {
                if (! visited[n.y][n.x]) {
                    n.dist = p.dist + 1;
                    q.addLast(n);
                    visited[n.y][n.x] = true;

                }
            }
        }
        return maxDist;
    }

    static List<Point> neighbors(Point p) {
        List<Point> list = new ArrayList<>();
        // (x-1, y), (x+1, y), (x, y-1), (x, y+1)
        if (p.x > 0) {
            if (isConnected(p.x, p.y, p.x-1, p.y)) {
                list.add(new Point(p.x-1, p.y));
            }
        }
        if (p.x < W-1) {
            if (isConnected(p.x, p.y, p.x+1, p.y)) {
                list.add(new Point(p.x+1, p.y));
            }
        }
        if (p.y > 0) {
            if (isConnected(p.x, p.y, p.x, p.y-1)) {
                list.add(new Point(p.x, p.y-1));
            }
        }
        if (p.y < H-1) {
            if (isConnected(p.x, p.y, p.x, p.y+1)) {
                list.add(new Point(p.x, p.y+1));
            }
        }
        return list;
    }

    static void findExits() {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                if (isExit(i, j)) {
                    exits.add(new Point(j, i));
                }
            }
        }
    }

    static void printExits() {
        for (Point p : exits) {
            System.out.println(p);
        }
    }

    static void printGraph() {
        for (int i = 0; i < H - 1; i++) {
            printRow(i);
            printInterRow(i);
        }
        printRow(H-1);
    }

    static void printInterRow(int i) {
        for (int j = 0; j < W; j++) {
            if (isConnected(j, i, j, i+1)) {
                System.out.print('|');
            } else {
                System.out.print(' ');
            }
            System.out.print(' ');
        }
        System.out.println();
    }

    static void printRow(int i) {
        for (int j = 0; j < W - 1; j++) {
            System.out.print('*');
            if (isConnected(j, i, j+1, i)) {
                System.out.print('-');
            } else {
                System.out.print(' ');
            }
        }
        System.out.println('*');
    }

    static boolean isConnected(int x1, int y1, int x2, int y2) {
        if (y1 == y2) {
            if (x1 - x2 == 1) {
                if (maze[y1 * 2 + 1][2 * x2 + 2] == ' ') {
                    return true;
                }
            }
            if (x2 - x1 == 1) {
                if (maze[y1 * 2 + 1][2 * x1 + 2] == ' ') {
                    return true;
                }
            }
        }
        if (x1 == x2) {
            if (y1 - y2 == 1) {
                if (maze[2 * y2 + 2][x1 * 2 + 1] == ' ') {
                    return true;
                }
            }
            if (y2 - y1 == 1) {
                if (maze[2 * y1 + 2][x1 * 2 + 1] == ' ') {
                    return true;
                }
            }
        }
        return false;
    }

    static boolean isExit(int i, int j) {
        return (i == 0 && maze[0][j*2+1] == ' ')
            || (i == H-1 && maze[2*H][j*2+1] == ' ')
            || (j == 0 && maze[i*2+1][0] == ' ')
            || (j == W-1 && maze[i*2+1][2*W] == ' ');
    }

    static void printMaze() {
        for (int i = 0; i < 2*H+1; i++) {
            for (int j = 0; j < 2*W+1; j++) {
                System.out.print(maze[i][j]);
            }
            System.out.println();
        }
    }

    static class Point {
        public int x;
        public int y;
        public int dist;
        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }
        public Point(int x, int y, int dist) {
            this(x, y);
            this.dist = dist;
        }
        public String toString() {
            return String.format("(%d,%d)", x, y);
        }
    }
}
