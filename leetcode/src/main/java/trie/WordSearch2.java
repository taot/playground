/**
 * LeetCode
 *
 * Problem 212: Word Search II
 */

package trie;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

public class WordSearch2 {

    static public boolean find(char[][] board, boolean[][] visited, int i, int j, String word, int k) {
        if (k == word.length()) {
            return true;
        }

        char c = word.charAt(k);
        boolean res = false;

        if (i > 0 && board[i-1][j] == c && ! visited[i-1][j]) {
            visited[i-1][j] = true;
            res = find(board, visited, i-1, j, word, k+1);
            visited[i-1][j] = false;
        }
        if (res) {
            return true;
        }

        if (i < board.length-1 && board[i+1][j] == c && ! visited[i+1][j]) {
            visited[i+1][j] = true;
            res = find(board, visited, i+1, j, word, k+1);
            visited[i+1][j] = false;
        }
        if (res) {
            return true;
        }

        if (j > 0 && board[i][j-1] == c && ! visited[i][j-1]) {
            visited[i][j-1] = true;
            res = find(board, visited, i, j-1, word, k+1);
            visited[i][j-1] = false;
        }
        if (res) {
            return true;
        }

        if (j < board[0].length-1 && board[i][j+1] == c && ! visited[i][j+1]) {
            visited[i][j+1] = true;
            res = find(board, visited, i, j+1, word, k+1);
            visited[i][j+1] = false;
        }
        if (res) {
            return true;
        }

        return false;
    }

    static public List<String> findWords(char[][] board, String[] words) {
        Set<String> output = new TreeSet<>();
        for (String w : words) {
            boolean res = false;
            boolean[][] visited = new boolean[board.length][board[0].length];
            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j < board[0].length; j++) {
                    if (board[i][j] == w.charAt(0)) {
                        visited[i][j] = true;
                        res |= find(board, visited, i, j, w, 1);
                        visited[i][j] = false;
                    }
                }
            }
            if (res) {
                output.add(w);
            }
        }

        return new ArrayList(output);
    }
}
