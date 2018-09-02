package trie;

public class MaximumXOROfTwoNumbersInAnArray {

    static class TrieNode {
        int val;
        Integer num;
        TrieNode[] next = new TrieNode[2];
    }

    public void insert(TrieNode root, int n) {
        TrieNode node = root;
        for (int i = 30; i >= 0; i--) {
            int x = (n >> i) & 1;
            if (node.next[x] == null) {
                node.next[x] = new TrieNode();
            }
            node = node.next[x];
        }
        node.num = n;
    }

    public int find(TrieNode root, int n) {
        TrieNode node = root;
        for (int i = 30; i >= 0; i--) {
            int x = ((n >> i) & 1);
            if (x == 0) {
                x = 1;
            } else {
                x = 0;
            }
            if (node.next[x] != null) {
                node = node.next[x];
            } else if (node.next[x ^ 1] != null) {
                node = node.next[x ^ 1];
            } else {
                return -1;
            }
        }
        return node.num;
    }

    public int findMaximumXOR(int[] nums) {
        TrieNode root = new TrieNode();
        for (int n : nums) {
            insert(root, n);
        }

        int max = -1;
        for (int n : nums) {
            int m = find(root, n);
            if (m >= 0) {
                int xor = n ^ m;
                if (xor > max) {
                    max = xor;
                }
            }
        }

        return max;
    }
}
