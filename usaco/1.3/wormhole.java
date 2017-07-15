/*
ID: libra_k1
LANG: JAVA
TASK: wormhole
*/
import java.io.*;
import java.util.*;

class wormhole {

    private static String task = "wormhole";

    private static int N;

    private static int trapCount = 0;

    private static Map<Integer, List<Coord>> rows = new HashMap<Integer, List<Coord>>();

    private static Map<Integer, Coord> pointsById = new HashMap<Integer, Coord>();

    public static void main1(String[] args) {
        N = 6;
        int[] arr = new int[N];
        for (int i = 0; i < N; i++) {
            arr[i] = i + 1;
        }
        permute(arr, 0);
    }


    private static void permute(int[] arr, int idx) {
        if (idx >= arr.length) {
            // printArr(arr);
            boolean trapped = checkTrap(arr);
            if (trapped) {
                printArr(arr);
                trapCount++;
            }
            return;
        }

        for (int i = 0; i <= idx; i++) {
            swap(arr, idx, i);
            permute(arr, idx + 2);
            swap(arr, idx, i);
        }
    }

    private static void swap(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    private static void printArr(int[] arr) {
        boolean first = true;
        for (int i = 0; i < arr.length; i++) {
            if (first) {
                first = false;
            } else {
                // System.out.print(',');
            }
            // System.out.print(arr[i]);
        }
        // System.out.println();
    }

    private static boolean checkTrap(int[] arr) {
        setupPairs(arr);
        for (Coord p : pointsById.values()) {
            boolean t = walk(p);
            if (t) {
                return true;
            }
        }
        return false;
    }

    public static void main(String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        Coord[] holes = new Coord[N];

        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            int x = Integer.parseInt(st.nextToken());
            int y = Integer.parseInt(st.nextToken());
            int id = i + 1;
            Coord c = new Coord(id, x, y);
            insertToRows(c);
            pointsById.put(id, c);
        }

        preprocess();

        for (int row : rows.keySet()) {
            List<Coord> list = rows.get(row);
            // System.out.println("row " + row);
            // System.out.println(list);
        }

        int[] arr = new int[N];
        for (int i = 0; i < N; i++) {
            arr[i] = i + 1;
        }
        permute(arr, 0);

        // System.out.println(trapCount);
        out.println(trapCount);

        out.close();
    }

    private static void setupPairs(int[] arr) {
        for (int i = 0; i < N / 2; i++) {
            Coord p1 = pointsById.get(arr[i * 2]);
            Coord p2 = pointsById.get(arr[i * 2 + 1]);
            p1.pair = p2;
            p2.pair = p1;
        }
    }

    /**
     * If trapped, return true
     */
    private static boolean walk(Coord start) {
        BitSet visited = new BitSet();
        Map<Integer, Integer> visitCount = new HashMap<Integer, Integer>();
        boolean transported = false;
        Coord p = start;

        do {
            visited.set(p.id);
            incCount(visitCount, p.id);
            if (transported) {
                p = p.next;
            } else {
                p = p.pair;
            }
            transported = ! transported;

        } while (p != null && getVisitCount(visitCount, p.id) <= 2);
        // } while (p != null && (! visited.get(p.id) || transported));

        boolean trapped = p != null;
        return trapped;
    }

    private static int getVisitCount(Map<Integer, Integer> visitCount, int id) {
        Integer i = visitCount.get(id);
        if (i == null) {
            return 0;
        } else {
            return i;
        }
    }

    private static void incCount(Map<Integer, Integer> visitCount, int id) {
        Integer i = visitCount.get(id);
        if (i == null) {
            visitCount.put(id, 1);
        } else {
            visitCount.put(id, i + 1);
        }
    }

    private static void insertToRows(Coord c) {
        List<Coord> row = rows.get(c.y);
        if (row == null) {
            row = new ArrayList<Coord>();
            rows.put(c.y, row);
        }
        row.add(c);
    }

    private static void preprocess() {
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
        public String toString() {
            return String.format("%d (%d, %d)", id, x, y);
        }
    }
}
