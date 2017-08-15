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
    static List[][] graph;
    static boolean[][] exits;

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

        printMaze();

        out.close();
    }

    static void buildGraph() {
        graph = new List[H][W];
        exits = new boolean[H][W];
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                if (isExit(i, j)) {
                    exits[i][j] = true;
                }
            }
            
        }
    }

    static boolean isExit(int i, int j) {
        return (i == 0 && maze[0][j*2+1] == ' ')
            || (i == H-1 && maze[2*H][j*2+1] == ' ')
            || (j == 0 && maze[i*2+1][0] == ' ')
            || (j == W-1 && maze[i*2+1][2*W]);
    }

    static void printMaze() {
        for (int i = 0; i < 2*H+1; i++) {
            for (int j = 0; j < 2*W+1; j++) {
                System.out.print(maze[i][j]);
            }
            System.out.println();
        }
    }
}
