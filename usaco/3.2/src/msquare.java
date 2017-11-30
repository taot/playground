/*
ID: libra_k1
LANG: JAVA
TASK: msquare
*/
import java.io.*;
import java.util.*;

class msquare {

    private static String task = "msquare";

    static int[] srcData = {1, 2, 3, 4, 5, 6, 7, 8};

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());

        int[] tgtData = new int[8];
        for (int i = 0; i < 8; i++) {
            tgtData[i] = Integer.parseInt(st.nextToken());
        }

        MSquare src = new MSquare();
        src.setData(srcData);
        MSquare tgt = new MSquare();
        tgt.setData(tgtData);

        MSquare res = bfs(src, tgt);

        if (res == null) {
            System.out.println("NONE");
        } else {
            StringBuilder sb = new StringBuilder();
            MSquare s = res;
            int count = 0;
            while (s.parent != null) {
                sb.append(s.action);
                s = s.parent;
                count++;
            }
            System.out.println(count);
            out.println(count);
            String path = sb.reverse().toString();
//            path = "0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345";
            for (int i = 0; i < ((path.length() - 1) / 60) + 1; i++) {
                String subs = path.substring(i*60, (i+1)*60 > path.length() ? path.length() : (i+1) * 60);
                System.out.println(subs);
                out.println(subs);
            }
        }

        out.close();
    }

    static MSquare bfs(MSquare src, MSquare tgt) {
        Deque<MSquare> q = new ArrayDeque<>();
        Set<MSquare> visited = new HashSet<>();
        if (src.equals(tgt)) {
            return src;
        }
        q.addFirst(src);
        visited.add(src);
        MSquare s;
        MSquare res = null;
        while ((s = q.pollLast()) != null && res == null) {
            MSquare a = s.A();
            MSquare b = s.B();
            MSquare c = s.C();
            MSquare[] arr = { a, b, c };
            for (MSquare x : arr) {
                if (visited.contains(x)) {
                    continue;
                }
                if (tgt.equals(x)) {
                    res = x;
                    break;
                }
                q.addFirst(x);
                visited.add(x);
            }
        }

        return res;
    }

    static class MSquare {

        public int[] row1 = new int[4];
        public int[] row2 = new int[4];

        public MSquare parent = null;
        public char action = ' ';

        public void setData(int[] data) {
            row1[0] = data[0];
            row1[1] = data[1];
            row1[2] = data[2];
            row1[3] = data[3];
            row2[0] = data[7];
            row2[1] = data[6];
            row2[2] = data[5];
            row2[3] = data[4];
        }

        public MSquare copy() {
            MSquare s = new MSquare();
            System.arraycopy(this.row1, 0, s.row1, 0, 4);
            System.arraycopy(this.row2, 0, s.row2, 0, 4);
            s.parent = this;
            return s;
        }

        public MSquare A() {
            MSquare s = this.copy();
            int[] tmp = s.row1;
            s.row1 = s.row2;
            s.row2 = tmp;
            s.action = 'A';
            return s;
        }

        public MSquare B() {
            MSquare s = this.copy();
            rshift(s.row1);
            rshift(s.row2);
            s.action = 'B';
            return s;
        }

        public MSquare C() {
            MSquare s = this.copy();
            int tmp = s.row1[1];
            s.row1[1] = s.row2[1];
            s.row2[1] = s.row2[2];
            s.row2[2] = s.row1[2];
            s.row1[2] = tmp;
            s.action = 'C';
            return s;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            MSquare mSquare = (MSquare) o;

            if (!Arrays.equals(row1, mSquare.row1)) return false;
            return Arrays.equals(row2, mSquare.row2);
        }

        @Override
        public int hashCode() {
            int result = Arrays.hashCode(row1);
            result = 31 * result + Arrays.hashCode(row2);
            return result;
        }

        private void rshift(int[] a) {
            int tmp = a[a.length - 1];
            for (int i = a.length - 1; i >= 1; i--) {
                a[i] = a[i-1];
            }
            a[0] = tmp;
        }
    }
}
