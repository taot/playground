package trie;

import org.junit.Assert;
import org.junit.Test;

public class AddAndSearchWordTest {

    @Test
    public void test_example_1() {
        WordDictionary dict = new WordDictionary();
        dict.addWord("bad");
        dict.addWord("dad");
        dict.addWord("mad");

        Assert.assertFalse(dict.search("pad"));
        Assert.assertTrue(dict.search("bad"));
        Assert.assertTrue(dict.search(".ad"));
        Assert.assertTrue(dict.search("b.."));
    }

    @Test
    public void test_my_1() {
        WordDictionary dict = new WordDictionary();
        dict.addWord("bad");
        dict.addWord("dad");
        dict.addWord("mad");
        dict.addWord("b");

        Assert.assertFalse(dict.search("bada"));
        Assert.assertTrue(dict.search("..."));
        Assert.assertTrue(dict.search("b"));
        Assert.assertTrue(dict.search("."));
    }

    @Test
    public void test_my_2() {
        WordDictionary dict = new WordDictionary();
        dict.addWord("abcdefghijklmnopqrstuvwxyz");


        Assert.assertTrue(dict.search(".........................."));
        Assert.assertTrue(dict.search("abcdefghijklmnopqrstuvwxyz"));
        Assert.assertFalse(dict.search("abcdefghijklmnopqrstuvwxy"));
    }
}
