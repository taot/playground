package string;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

public class LetterCombinationsPhoneNumberTest {

    @Test
    public void test_example_1() {
        List<String> list = new LetterCombinationsPhoneNumber().letterCombinations("23");
        Set<String> expected = new TreeSet<>(Arrays.asList(new String[] { "ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf" }));
        Assert.assertEquals(expected, new TreeSet<String>(list));
    }

    @Test
    public void test_my_1() {
        List<String> list = new LetterCombinationsPhoneNumber().letterCombinations("");
        Assert.assertTrue(list.isEmpty());
    }

    @Test
    public void test_my_2() {
        List<String> list = new LetterCombinationsPhoneNumber().letterCombinations("2");
        Set<String> expected = new TreeSet<>(Arrays.asList(new String[] { "a", "b", "c" }));
        Assert.assertEquals(expected, new TreeSet<String>(list));
    }
}
