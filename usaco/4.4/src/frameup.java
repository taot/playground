/*
ID: libra_k1
LANG: JAVA
TASK: frameup
*/
import java.io.*;
import java.util.*;

class frameup {

    private static String task = "frameup";

    static int H, W;
    static char[][] array;
    static int N = 26;

    static int[] widths = new int[N];
    static int[] heights = new int[N];
    static int[] tops = new int[N];
    static int[] lefts = new int[N];
    static char[] frames = new char[N];
    static int frameCount = 0;
    static boolean[] visited = new boolean[N];
    static PrintWriter out;
    static List<String> results = new ArrayList<>();

    public static void main (String [] args) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(task + ".in"));
        out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        read(reader);

        findFrames();

        char[][] plate = new char[H][W];
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                plate[i][j] = '.';
            }
        }

        char[] seq = new char[N];
        dfs(plate, seq, 0);

        Collections.sort(results);
        for (String s : results) {
            System.out.println(s);
            out.println(s);
        }

        out.close();
    }

    static void findFrames() {
        // find frames
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                char c = array[i][j];
                if (c == '.') {
                    continue;
                }
                boolean found = false;
                for (int k = 0; k < frameCount; k++) {
                    if (c == frames[k]) {
                        found = true;
                        break;
                    }
                }
                if (! found) {
                    frames[frameCount] = c;
                    frameCount++;
                }
            }
        }

        // find widths and heights
        for (int k = 0; k < frameCount; k++) {
            char c = frames[k];
            int left, right, top, bottom;
            left = top = Integer.MAX_VALUE;
            right = bottom = -1;
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    if (array[i][j] != c) {
                        continue;
                    }
                    if (i < top) {
                        top = i;
                    }
                    if (i > bottom) {
                        bottom = i;
                    }
                    if (j < left) {
                        left = j;
                    }
                    if (j > right) {
                        right = j;
                    }
                }
            }
            widths[k] = right - left + 1;
            heights[k] = bottom - top + 1;
            tops[k] = top;
            lefts[k] = left;
        }
    }

    static void dfs(char[][] plate, char[] seq, int pos) {
        if (pos >= frameCount) {
            if (isValid(plate)) {
                StringBuilder sb = new StringBuilder();
                for (int k = frameCount - 1; k >= 0; k--) {
                    sb.append(seq[k]);
                }
                results.add(sb.toString());
            }
        }
        for (int i = 0; i < frameCount; i++) {
            if (visited[i]) {
                continue;
            }
            seq[pos] = frames[i];
            apply(plate, i, lefts[i], tops[i]);
            if (isPossible(plate)) {
                visited[i] = true;
                dfs(plate, seq, pos + 1);
                visited[i] = false;
            }
            unapply(plate, i, lefts[i], tops[i]);
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

    static void printArray(int[] a, int length) {
        for (int i = 0; i < length; i++) {
            System.out.print(a[i]);
        }
        System.out.println();
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
