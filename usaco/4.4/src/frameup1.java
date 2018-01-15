/*
ID: libra_k1
LANG: JAVA
TASK: frameup
*/

import java.io.*;
import java.util.StringTokenizer;

class frameup1 {

    private static String task = "frameup";

    static int H, W;
    static char[][] array;

    static int[] widths = new int[] { 6, 6, 4, 4, 3 };
    static int[] heights = new int[] { 8, 5, 4, 5, 4 };
    static char[] frames = new char[] { 'E', 'D', 'A', 'B', 'C' };
    static boolean[] visited = new boolean[5];
    static PrintWriter out;

    public static void main (String [] args) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(task + ".in"));
        out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        read(reader);

        char[][] plate = new char[H][W];
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                plate[i][j] = '.';
            }
        }
        char[] seq = new char[5];
        dfs(plate, seq, 0);


        out.close();
    }

    static void dfs(char[][] plate, char[] seq, int pos) {
        if (pos >= 5) {
            if (isValid(plate)) {
                for (int k = 4; k >= 0; k--) {
                    System.out.print(seq[k]);
                    out.print(seq[k]);
                }
                System.out.println();
                out.println();
            }
        }
        for (int i = 0; i < 5; i++) {
            if (visited[i]) {
                continue;
            }
            for (int t = 0; t < H - heights[i] + 1; t++) {
                for (int l = 0; l < W - widths[i] + 1; l++) {
                    seq[pos] = frames[i];
                    apply(plate, i, l, t);
                    if (isPossible(plate)) {
                        visited[i] = true;
                        dfs(plate, seq, pos + 1);
                        visited[i] = false;
                    }
                    unapply(plate, i, l, t);
                }
            }
        }
    }

    static void apply(char[][] plate, int i, int left, int top) {
        applyImpl(plate, i, left, top, '.', frames[i]);
    }

    static void unapply(char[][] plate, int i, int left, int top) {
        applyImpl(plate, i, left, top, frames[i], '.');
    }

    static void applyImpl(char[][] plate, int i, int left, int top, char from, char to) {
        int w = widths[i];
        int h = heights[i];

        for (int k = left; k < left + w; k++) {
            if (plate[top][k] == from) {
                plate[top][k] = to;
            }
            if (plate[top + h - 1][k] == from) {
                plate[top + h - 1][k] = to;
            }
        }
        for (int k = top; k < top + h; k++) {
            if (plate[k][left] == from) {
                plate[k][left] = to;
            }
            if (plate[k][left + w - 1] == from) {
                plate[k][left + w - 1] = to;
            }
        }
    }

    static boolean isPossible(char[][] plate) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                if (plate[h][w] != '.' && array[h][w] != '.' && plate[h][w] != array[h][w]) {
                    return false;
                }
            }
        }
        return true;
    }

    static boolean isValid(char[][] plate) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                if (plate[h][w] != array[h][w]) {
                    return false;
                }
            }
        }
        return true;
    }

    static void read(BufferedReader reader) throws IOException {
        StringTokenizer st = new StringTokenizer(reader.readLine());
        H = Integer.parseInt(st.nextToken());
        W = Integer.parseInt(st.nextToken());
        array = new char[H][W];
        for (int i = 0; i < H; i++) {
            String l = reader.readLine();
            char[] chars = l.toCharArray();
            System.arraycopy(chars, 0, array[i], 0, chars.length);
        }
    }
}
