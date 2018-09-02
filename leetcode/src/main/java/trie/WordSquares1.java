/**
 * LeetCode
 *
 * Problem 425: Word Squares
 */

package trie;

import java.util.*;

public class WordSquares1 {

    class TrieNode {
        TrieNode[] next = new TrieNode[26];
        String word = null;
    }

    TrieNode makeTrie(String[] words) {
        TrieNode root = new TrieNode();
        for (String w : words) {
            TrieNode node = root;
            for (char c : w.toCharArray()) {
                if (node.next[c - 'a'] == null) {
                    node.next[c - 'a'] = new TrieNode();
                }
                node = node.next[c - 'a'];
            }
            node.word = w;
        }
        return root;
    }

    List<List<String>> results = new ArrayList<>();

    char[][] square;

    Map<Character, List<Integer>> map = new HashMap<>();

    boolean check(String[] words, int[] selection) {
        for (int i = 0; i < selection.length; i++) {
            String w = words[selection[i]];
            for (int j = 0; j < selection.length; j++) {
                square[i][j] = w.charAt(j);
            }
        }
        for (int i = 0; i < selection.length; i++) {
            for (int j = 0; j < selection.length; j++) {
                if (square[i][j] != square[j][i]) {
                    return false;
                }
            }
        }
        return true;
    }

    void dfs(String[] words, String first, int k, int[] selection) {
        if (k == selection.length) {
            if (check(words, selection)) {
                List<String> r = new ArrayList<>();
                for (int s : selection) {
                    r.add(words[s]);
                }
                results.add(r);
            }
            return;
        }

        List<Integer> candidates = map.get(first.charAt(k));

        if (candidates == null) {
            return;
        }

        for (int i : candidates) {
            selection[k] = i;
            dfs(words, first, k+1, selection);
        }
    }

    public List<List<String>> wordSquares(String[] words) {
        int N = words.length;
        int M = words[0].length();
        square = new char[M][M];
        results.clear();
        // TrieNode root = makeTrie(words);

        // try using map
        map.clear();
        for (int i = 0; i < words.length; i++) {
            String w = words[i];
            char c = w.charAt(0);
            List<Integer> list = map.get(c);
            if (list == null) {
                list = new ArrayList<>();
                map.put(c, list);
            }
            list.add(i);
        }

        int[] selection = new int[M];

        for (int i = 0; i < N; i++) {
            String w = words[i];
            dfs(words, w, 0, selection);
        }

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
