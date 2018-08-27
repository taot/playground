package hashtable;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static hashtable.TopKFrequentWords.*;

public class TopKFrequentWordsTest {

    @Test
    public void test_example_1() {
        String[] input = {"i", "love", "leetcode", "i", "love", "coding"};
        List<String> actual = topKFrequent(input, 2);
        List<String> expected = Arrays.asList("i", "love");
        Assert.assertEquals(expected, actual);
    }

    @Test
    public void test_example_2() {
        String[] input = {"the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"};
        List<String> actual = topKFrequent(input, 4);
        List<String> expected = Arrays.asList("the", "is", "sunny", "day");
        Assert.assertEquals(expected, actual);
    }

    @Test
    public void test_my_1() {
        String[] input = {"1", "1", "2", "2", "2", "3", "3", "3", "4", "4", "4"};
        List<String> actual = topKFrequent(input, 2);
        List<String> expected = Arrays.asList("2", "3");
        Assert.assertEquals(expected, actual);
    }
}
