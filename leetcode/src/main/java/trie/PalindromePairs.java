/**
 * LeetCode
 *
 * Problem 336: Palindrome Pairs
 */

package trie;

import java.util.*;

public class PalindromePairs {

    class TrieNode {
        TrieNode[] next = new TrieNode[26];
        Integer idx = null;
    }

    TrieNode makeTrie(String[] words) {
        TrieNode root = new TrieNode();
        for (int i = 0; i < words.length; i++) {
            String w = words[i];
            if (w.length() == 0) {
                root.idx = i;
                continue;
            }
            TrieNode node = root;
            for (int j = 0; j < w.length(); j++) {
                char c = w.charAt(j);
                if (node.next[c - 'a'] == null) {
                    node.next[c - 'a'] = new TrieNode();
                }
                node = node.next[c - 'a'];
                if (j == w.length() - 1) {
                    node.idx = i;
                }
            }
        }
        return root;
    }

    boolean isPalindromePrefix(String word, int i) {
        for (int j = 0; j < i / 2; j++) {
            if (word.charAt(j) != word.charAt(i - j - 1)) {
                return false;
            }
        }
        return true;
    }

    boolean isPalindrome(StringBuilder buf) {
        int len = buf.length();
        for (int j = 0; j < len / 2; j++) {
            if (buf.charAt(j) != buf.charAt(len - j - 1)) {
                return false;
            }
        }
        return true;
    }

    void getPalindromePostfix(TrieNode node, List<Integer> results, StringBuilder buf) {
        if (node.idx != null && isPalindrome(buf)) {
            results.add(node.idx);
        }
        for (char c = 'a'; c <= 'z'; c++) {
            TrieNode next = node.next[c - 'a'];
            if (next != null) {
                buf.append(c);
                getPalindromePostfix(next, results, buf);
                buf.deleteCharAt(buf.length() - 1);
            }
        }
    }

    public List<List<Integer>> palindromePairs(String[] words) {
        TrieNode root = makeTrie(words);

        List<List<Integer>> results = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            String w = words[i];
            TrieNode node = root;
            int j = w.length();
            while (j > 0 && node != null) {

                if (node.idx != null && isPalindromePrefix(w, j) && node.idx != i) {
                    results.add(Arrays.asList(node.idx, i));
                }

                char c = w.charAt(j-1);
                node = node.next[c - 'a'];
                j--;
            }

            if (j <= 0 && node != null) {
                List<Integer> postfixes = new ArrayList<>();
                StringBuilder buf = new StringBuilder();
                getPalindromePostfix(node, postfixes, buf);
                for (Integer k : postfixes) {
                    if (k == i) {
                        continue;
                    }
                    results.add(Arrays.asList(k, i));
                }
            }
        }

        Collections.sort(results, new Comparator<List<Integer>>() {
            @Override
            public int compare(List<Integer> o1, List<Integer> o2) {
                int n1 = o1.get(0);
                int n2 = o2.get(0);
                if (n1 != n2) {
                    return n1 - n2;
                }
                n1 = o1.get(1);
                n2 = o2.get(1);
                return n1 - n2;
            }
        });

        return results;
    }
}
