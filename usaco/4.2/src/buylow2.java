/*
ID: libra_k1
LANG: JAVA
TASK: buylow
*/
import java.io.*;
import java.util.*;

class buylow2 {

    private static String task = "buylow";

    static int N;
    static int[] series;
    static int[] lengths;
    static List<Integer>[] prev;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in7"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        series = new int[N+1];
        lengths = new int[N+1];
        prev = new List[N+1];
        st = new StringTokenizer(f.readLine());
        for (int i = 0; i < N; i++) {
            if (! st.hasMoreElements()) {
                st = new StringTokenizer(f.readLine());
            }
            series[i] = Integer.parseInt(st.nextToken());

        }
        series[N] = 0;
        for (int i = 0; i < N + 1; i++) {
            prev[i] = new ArrayList<>();
        }

        dp();

        printArr(series);
        printArr(lengths);

        int len = lengths[N] - 1;
        System.out.println(len);

        Node tree = createTree(N);
//        Node tree2 = mergeTree(tree);

        int nLeaf = countLeaf(tree);
        System.out.println(nLeaf);

        out.close();
    }

    static int countLeaf(Node t) {
        if (t.children.isEmpty()) {
            return 1;
        }
        int sum = 0;
        for (Node c : t.children) {
            sum += countLeaf(c);
        }
        return sum;
    }

    static Node createTree(int idx) {
        Node n = new Node(series[idx]);
        for (int i : prev[idx]) {
            Node c = createTree(i);
            n.children.add(c);
        }
        return n;
    }

    static Node mergeTree(Node t) {
        Node t2 = new Node(t.v);
        t2.children.addAll(mergeChildren(t.children));
        return t2;
    }

    static List<Node> mergeChildren(List<Node> children) {
        Map<Integer, List<Node>> groups = groupBy(children);
        List<Node> children2 = new ArrayList<>();
        for (Map.Entry<Integer, List<Node>> ent : groups.entrySet()) {
            List<Node> ch = new ArrayList<>();
            for (Node n : ent.getValue()) {
                ch.addAll(n.children);
            }
            List<Node> ch2 = mergeChildren(ch);
            Node t = new Node(ent.getKey());
            t.children.addAll(ch2);
            children2.add(t);
        }
        return children2;
    }

    static Map<Integer, List<Node>> groupBy(List<Node> trees) {
        Map<Integer, List<Node>> groups = new HashMap<>();
        for (Node t : trees) {
            List<Node> list = groups.get(t.v);
            if (list == null) {
                list = new ArrayList<>();
                groups.put(t.v, list);
            }
            list.add(t);
        }
        return groups;
    }

    static void dp() {
        for (int i = 0; i < N + 1; i++) {
            int m = 0;
            for (int j = i - 1; j >= 0; j--) {
                if (series[i] >= series[j]) {
                    continue;
                }
                if (lengths[j] > m) {
                    m = lengths[j];
                    prev[i].clear();
                    prev[i].add(j);
                } else if (lengths[j] == m) {
                    prev[i].add(j);
                }
            }
            lengths[i] = m + 1;
        }
    }

    static void printArr(int[] a) {
        for (int i = 0; i < a.length; i++) {
            System.out.print(padding(a[i]) + " ");
        }
        System.out.println();
    }

    static String padding(int x) {
        String s = String.valueOf(x);
        int l = s.length();
        for (int i = 0; i < 3 - l; i++) {
            s = " " + s;
        }
        return s;
    }

    static class Node {
        int v;
        List<Node> children;

        public Node(int v) {
            this.v = v;
            this.children = new ArrayList<>();
        }
    }
}
