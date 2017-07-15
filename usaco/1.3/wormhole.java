/*
ID: libra_k1
LANG: JAVA
TASK: wormhole
*/
import java.io.*;
import java.util.*;

class template {

    private static String task = "wormhole";

    private static int N;

    private static Map<Integer, List<Coord>> rows = new HashMap<Integer, List<Coord>>();

    private static Map<Integer, Coord> pointsById = new HashMap<Integer, Coord>();

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        Coord[] holes = new Coord[N];
        int id = 0;
        for (int i = 0; i < N; i++) {
            StringTokenizer st = new StringTokenizer(f.readLine());
            int x = Integer.parseInt(st.nextToken());
            int y = Integer.parseInt(st.nextToken());
            Coord c = new Coord(i, x, y);
            insertToRows(c);
            pointsById.put(i, c);
        }





        out.println(i1+i2);
        out.close();
    }

    

    /**
     * If trapped, return true
     */
    private static boolean walk(Coord start) {
        BitSet visited = new BitSet();
        Coord p = start;
        visited.set(p.id);
        p = p.next;
        while (p != null && ! visited.get(p.id)) {
            p = p.next;
        }
        boolean trapped = p != null;
        return trapped;
    }

    private static void insertToRows(Coord c) {
        List<Coord> row = rows.get(c.y);
        if (row == null) {
            row = new ArrayList<Coord>();
            rows.put(c.y, row);
        }
        row.add(c);
    }

    private static preprocess() {
        for (List<Coord> row : rows.values()) {
            // sort by x
            Collections.sort(row, new Comparator<Coord>() {
                public int compare(Coord c1, Coord c2) {
                    return c1.x - c2.x;
                }
            });
            // link next
            for (int i = 0; i < row.size() - 1; i++) {
                row.get(i).next = row.get(i + 1);
            }
        }
    }

    private static class Coord {
        public final int x;
        public final int y;
        public final int id;
        public Coord next = null;
        public Coord pair = null;
        public Coord(int id, int x, int y) {
            this.id = id;
            this.x = x;
            this.y = y;
        }
    }
}
