package hashtable;

import org.junit.Assert;
import org.junit.Test;

import static hashtable.SortCharactersByFrequency.*;

public class SortCharactersByFrequencyTest {

    @Test
    public void test_example_1() {
        Assert.assertEquals("eert", frequencySort("tree"));
    }

    @Test
    public void test_example_2() {
        Assert.assertEquals("aaaccc", frequencySort("cccaaa"));
    }

    @Test
    public void test_my_1() {
        Assert.assertEquals("", frequencySort(""));
    }
}
