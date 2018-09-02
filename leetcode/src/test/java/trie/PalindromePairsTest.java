package trie;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class PalindromePairsTest {

    @Test
    public void test_example_1() {
        String[] input = {"abcd","dcba","lls","s","sssll"};
        List<List<Integer>> output = new PalindromePairs().palindromePairs(input);
        List<List<Integer>> expected = new ArrayList<>();
        expected.add(Arrays.asList(0, 1));
        expected.add(Arrays.asList(1, 0));
        expected.add(Arrays.asList(2, 4));
        expected.add(Arrays.asList(3, 2));
        Assert.assertEquals(expected, output);
    }

    @Test
    public void test_example_2() {
        String[] input = {"bat","tab","cat"};
        List<List<Integer>> output = new PalindromePairs().palindromePairs(input);
        List<List<Integer>> expected = new ArrayList<>();
        expected.add(Arrays.asList(0, 1));
        expected.add(Arrays.asList(1, 0));
        Assert.assertEquals(expected, output);
    }

    @Test
    public void test_my_1() {
        String[] input = {"", "abba"};
        List<List<Integer>> output = new PalindromePairs().palindromePairs(input);
        List<List<Integer>> expected = new ArrayList<>();
        expected.add(Arrays.asList(0, 1));
        expected.add(Arrays.asList(1, 0));
        Assert.assertEquals(expected, output);
    }

    @Test
    public void test_my_2() {
        String[] input = {"abc", "ba"};
        List<List<Integer>> output = new PalindromePairs().palindromePairs(input);
        List<List<Integer>> expected = new ArrayList<>();
        expected.add(Arrays.asList(0, 1));
        Assert.assertEquals(expected, output);
    }

    @Test
    public void test_my_3() {
        String[] input = {""};
        List<List<Integer>> output = new PalindromePairs().palindromePairs(input);
        List<List<Integer>> expected = new ArrayList<>();
        Assert.assertEquals(expected, output);
    }
}
