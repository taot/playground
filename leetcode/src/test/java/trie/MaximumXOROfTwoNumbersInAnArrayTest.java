package trie;

import org.junit.*;

public class MaximumXOROfTwoNumbersInAnArrayTest {

    private MaximumXOROfTwoNumbersInAnArray obj = new MaximumXOROfTwoNumbersInAnArray();

    @Test
    public void test_example_1() {
        int[] input = new int[] {3, 10, 5, 25, 2, 8};
        int output = obj.findMaximumXOR(input);
        Assert.assertEquals(28, output);
    }
}
