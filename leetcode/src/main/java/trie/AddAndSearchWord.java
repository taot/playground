/**
 * LeetCode
 *
 * Problem 211: Add and Search Word
 */

package trie;

class WordDictionary {

    static class TrieNode {
        char c;
        TrieNode[] children;
        boolean flag;

        public TrieNode(char c) {
            this.c = c;
            this.children = new TrieNode[26];
            this.flag = false;
        }
    }

    private TrieNode root;

    public WordDictionary() {
        root = new TrieNode(' ');
    }

    public void addWord(String word) {
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            int idx = c - 'a';
            if (node.children[idx] == null) {
                node.children[idx] = new TrieNode(c);
            }
            if (i == word.length() - 1) {
                node.children[idx].flag = true;
            }
            node = node.children[idx];
        }
    }

    private boolean recursive(TrieNode rt, String word, int k) {
        if (rt == null) {
            return false;
        }
        if (k == word.length()) {
            return rt.flag;
        } else {
            char c = word.charAt(k);
            if (c == '.') {
                for (char d = 'a'; d <= 'z'; d++) {
                    int idx = d - 'a';
                    if (recursive(rt.children[idx], word, k+1)) {
                        return true;
                    }
                }
                return false;
            } else {
                int idx = c - 'a';
                return recursive(rt.children[idx], word, k+1);
            }
        }
    }

    public boolean search(String word) {
        return recursive(root, word, 0);
    }
}

public class AddAndSearchWord {
}
