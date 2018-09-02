package trie;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static trie.WordSearch2.*;

public class WordSearch2Test {

    @Test
    public void test_example_1_part() {
        char[][] board = {
                {'o','a','a','n'},
                {'e','t','a','e'},
                {'i','h','k','r'},
                {'i','f','l','v'}
        };

        boolean[][] visited = new boolean[4][4];

        boolean output = false;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                output |= find(board, visited, i, j, "oath", 0);
            }
        }

        Assert.assertTrue(output);

        output = false;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                output |= find(board, visited, i, j, "eat", 0);
            }
        }

        Assert.assertTrue(output);

        output = false;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                output |= find(board, visited, i, j, "pea", 0);
            }
        }

        Assert.assertFalse(output);

        output = false;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                output |= find(board, visited, i, j, "rain", 0);
            }
        }

        Assert.assertFalse(output);
    }

    @Test
    public void test_example_1() {
        char[][] board = {
                {'o', 'a', 'a', 'n'},
                {'e', 't', 'a', 'e'},
                {'i', 'h', 'k', 'r'},
                {'i', 'f', 'l', 'v'}
        };

        List<String> output = findWords(board, new String[] {"oath","pea","eat","rain"});
        Assert.assertEquals(Arrays.asList("oath", "eat"), output);
    }

    @Test
    public void test_wrong_1() {
        char[][] board = {
                {'a'}
        };

        List<String> output = findWords(board, new String[] {"a"});
        Assert.assertEquals(Arrays.asList("a"), output);
    }

    @Test
    public void test_wrong_2() {
        char[][] board = {
                {'a'}
        };

        List<String> output = findWords(board, new String[] {"a", "a"});
        Assert.assertEquals(Arrays.asList("a"), output);
    }

    @Test
    public void test_wrong_3() {
        char[][] board = {
                {'a', 'a'}
        };

        List<String> output = findWords(board, new String[] {"aaa"});
        Assert.assertEquals(Arrays.asList(), output);
    }
}
