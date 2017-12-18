/*
ID: libra_k1
LANG: JAVA
TASK: heritage
*/
import java.io.*;
import java.util.*;

class heritage {

    private static String task = "heritage";

    static char[] preArray;
    static char[] inArray;

    static PrintWriter out;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        String s = f.readLine();
        inArray = s.toCharArray();
        s = f.readLine();
        preArray = s.toCharArray();

        Node tree = makeTree(0, 0, preArray.length);
        postOrder(tree);
        out.println();

        out.close();
    }

    static Node makeTree(int preStart, int inStart, int len) {
        if (len <= 0) {
            return null;
        }
        Node node = new Node();
        char rootKey = preArray[preStart];
        node.key = rootKey;
        int inRootIdx = find(inArray, rootKey);
        int leftLen = inRootIdx - inStart;
        int rightLen = len - leftLen - 1;
        node.left = makeTree(preStart + 1, inStart, leftLen);
        node.right = makeTree(preStart + leftLen + 1, inRootIdx + 1, rightLen);
        return node;
    }

    static void postOrder(Node node) {
        if (node == null) {
            return;
        }
        postOrder(node.left);
        postOrder(node.right);
        System.out.print(node.key);
        out.print(node.key);
    }

    static int find(char[] a, char key) {
        for (int i = 0; i < a.length; i++) {
            if (key == a[i]) {
                return i;
            }
        }
        return -1;
    }

    static class Node {
        char key;
        Node left;
        Node right;
    }
}
