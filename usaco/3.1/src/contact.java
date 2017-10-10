/*
ID: libra_k1
LANG: JAVA
TASK: contact
*/
import java.io.*;
import java.util.*;


class contact {

    private static String task = "contact";

    static int A, B, N;
    static String seq;
    static char[] chars;
    static TrieNode root = new TrieNode();

    public static void main (String [] args) throws IOException {
        long start = System.currentTimeMillis();

        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintStream out = new PrintStream(new File(task + ".out"));
        StringTokenizer st = new StringTokenizer(f.readLine());
        A = Integer.parseInt(st.nextToken());
        B = Integer.parseInt(st.nextToken());
        N = Integer.parseInt(st.nextToken());

        StringBuilder sb = new StringBuilder();
        String s;
        while ((s = f.readLine()) != null) {
            sb.append(s);
        }
        chars = sb.toString().toCharArray();

        constructTree();
        sumUp(root);
        print(System.out);
        print(out);

        System.out.println("Duration: " + (System.currentTimeMillis() - start) + " ms");

        out.close();
        System.exit(0);
    }

    private static void print(PrintStream ps) {
        Map<Integer, List<String>> map = new TreeMap<>((Integer t1, Integer t2) -> -1 * t1.compareTo(t2) );
        collectForPrint(root.left, "0", map);
        collectForPrint(root.right, "1", map);

        int count = 0;
        for (Integer k : map.keySet()) {
            count++;
            if (count > N) {
                break;
            }
            ps.print(k);
            List<String> list = map.get(k);
            Collections.sort(list, (String s1, String s2) -> {
                if (s1.length() < s2.length()) {
                    return -1;
                } else if (s1.length() > s2.length()) {
                    return 1;
                } else {
                    return s1.compareTo(s2);
                }
            });

            for (int i = 0; i < list.size(); i++) {
                if (i % 6 == 0) {
                    ps.println();
                } else {
                    ps.print(" ");
                }
                ps.print(list.get(i));
            }
            ps.println();
        }
    }

    private static void collectForPrint(TrieNode node, String path, Map<Integer, List<String>> map) {
        if (node == null) {
            return;
        }
        collectForPrint(node.left, path + '0', map);
        collectForPrint(node.right, path + '1', map);
        if (path.length() < A || path.length() > B) {
            return;
        }
        List<String> list = map.get(node.sumCount);
        if (list == null) {
            list = new ArrayList<>();
            map.put(node.sumCount, list);
        }
        list.add(path);

    }

    private static int sumUp(TrieNode node) {
        if (node == null) {
            return 0;
        }
        int c1 = sumUp(node.left);
        int c2 = sumUp(node.right);
        node.sumCount = c1 + c2 + node.count;
        return node.sumCount;
    }

    private static void constructTree() {
        for (int i = 0; i < chars.length; i++) {
            int len = Math.min(B, chars.length - i);
            TrieNode node = root;
            for (int j = 0; j < len; j++) {
                if (chars[i + j] == '0') {
                    if (node.left == null) {
                        node.left = new TrieNode();
                    }
                    node = node.left;
                } else {
                    if (node.right == null) {
                        node.right = new TrieNode();
                    }
                    node = node.right;
                }
                if (j == len - 1) {
                    node.count++;
                }
            }
        }
    }

    static class TrieNode {
        TrieNode left;
        TrieNode right;
        int count;
        int sumCount;

        public TrieNode() {
            left = right = null;
            count = sumCount = 0;
        }
    }
}
