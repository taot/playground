package string;

import org.junit.Assert;
import org.junit.Test;

import static string.LongestPalindromicSubstring.*;

public class LongestPalindromicSubstringTest {

    @Test
    public void test_example_1() {
        String p = longestPalindrome("babad");
        Assert.assertTrue("bab".equals(p) || "aba".equals(p));
    }

    @Test
    public void test_example_2() {
        String p = longestPalindrome("cbbd");
        Assert.assertEquals("bb", p);
    }

    @Test
    public void test_my_data_1() {
        String p = longestPalindrome("abcdefghihgfedcba");
        Assert.assertEquals("abcdefghihgfedcba", p);
    }

    @Test
    public void test_my_data_2() {
        String p = longestPalindrome("abcdefghhgfedcba");
        Assert.assertEquals("abcdefghhgfedcba", p);
    }

    @Test
    public void test_my_data_3() {
        String p = longestPalindrome("ebcdefghihgfedcba");
        Assert.assertEquals("bcdefghihgfedcb", p);
    }

    @Test
    public void test_empty_input_1() {
        String p = longestPalindrome("");
        Assert.assertEquals("", p);
    }

    @Test
    public void test_wrong_answer_1() {
        String p = longestPalindrome("a");
        Assert.assertEquals("a", p);
    }
}
