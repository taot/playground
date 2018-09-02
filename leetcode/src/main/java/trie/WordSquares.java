/**
 * LeetCode
 *
 * Problem 425: Word Squares
 */

package trie;

import java.util.*;

public class WordSquares {

    class TrieNode {
        TrieNode[] next = new TrieNode[26];
        Integer word = null;
    }

    TrieNode makeTrie(String[] words) {
        TrieNode root = new TrieNode();
        for (int i = 0; i < words.length; i++) {
            TrieNode node = root;
            String w = words[i];
            for (char c : w.toCharArray()) {
                if (node.next[c - 'a'] == null) {
                    node.next[c - 'a'] = new TrieNode();
                }
                node = node.next[c - 'a'];
            }
            node.word = i;
        }
        return root;
    }

    void collectWords(TrieNode node, List<Integer> list) {
        if (node.word != null) {
            list.add(node.word);
        }
        for (char c = 'a'; c <= 'z'; c++) {
            if (node.next[c - 'a'] == null) {
                continue;
            }
            collectWords(node.next[c - 'a'], list);
        }
    }

    List<Integer> getWords(TrieNode root, char[] prefix, int n) {
        TrieNode node = root;
        for (int i = 0; i < n; i++) {
            char c = prefix[i];
            if (node.next[c - 'a'] == null) {
                return Collections.emptyList();
            }
            node = node.next[c - 'a'];
        }
        List<Integer> list = new ArrayList<>();
        collectWords(node, list);
        return list;
    }

    List<List<String>> results = new ArrayList<>();

    char[][] square;

    TrieNode trie;

    void dfs(String[] words, int M, int k, char[][] selection) {
        if (k == M) {
            List<String> r = new ArrayList<>();
            for (int i = 0; i < selection.length; i++) {
                r.add(new String(selection[i]));
            }
            results.add(r);
            return;
        }

        List<Integer> candidates = getWords(trie, selection[k], k);

        if (candidates == null) {
            return;
        }

        for (int i : candidates) {
            String word = words[i];
            for (int row = 0; row < word.length(); row++) {
                selection[row][k] = word.charAt(row);
            }
            dfs(words, M, k+1, selection);
        }
    }

    public List<List<String>> wordSquares(String[] words) {
        int N = words.length;
        int M = words[0].length();
        square = new char[M][M];
        results.clear();
        trie = makeTrie(words);

        char[][] selection = new char[M][M];

        dfs(words, M, 0, selection);

        results.sort(new Comparator<List<String>>() {
            @Override
            public int compare(List<String> o1, List<String> o2) {
                for (int i = 0; i < o1.size(); i++) {
                    String s1 = o1.get(i);
                    String s2 = o2.get(i);
                    int r = s1.compareTo(s2);
                    if (r != 0) {
                        return r;
                    }
                }
                return 0;
            }
        });

        return results;
    }
}
