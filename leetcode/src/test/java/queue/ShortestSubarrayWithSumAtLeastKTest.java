package queue;

import org.junit.Assert;
import org.junit.Test;

import static queue.ShortestSubarrayWithSumAtLeastK.*;

public class ShortestSubarrayWithSumAtLeastKTest {

    @Test
    public void test_example_1() {
        int[] input = {1};
        Assert.assertEquals(1, shortestSubarray(input, 1));
    }

    @Test
    public void test_example_2() {
        int[] input = {1,2};
        Assert.assertEquals(-1, shortestSubarray(input, 4));
    }

    @Test
    public void test_example_3() {
        int[] input = {2,-1,2};
        Assert.assertEquals(3, shortestSubarray(input, 3));
    }

    @Test
    public void test_wrong_1() {
        int[] input = {77,19,35,10,-14};
        Assert.assertEquals(1, shortestSubarray(input, 19));
    }
}
