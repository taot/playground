package heap;

import org.junit.Assert;
import org.junit.Test;

import static heap.SlidingWindowMaximum.*;

public class SlidingWindowMaximumTest {

    @Test
    public void test_example_1() {
        int[] nums = {1,3,-1,-3,5,3,6,7};
        int[] expected = {3,3,5,5,6,7};
        int[] actual = maxSlidingWindow(nums, 3);
        Assert.assertArrayEquals(expected, actual);
    }

    @Test
    public void test_wrong_1() {
        int[] nums = {1,2,3};
        int[] expected = {1,2,3};
        int[] actual = maxSlidingWindow(nums, 0);
        Assert.assertArrayEquals(expected, actual);
    }

    @Test
    public void test_my_1() {
        int[] nums = {};
        int[] expected = {};
        int[] actual = maxSlidingWindow(nums, 2);
        Assert.assertArrayEquals(expected, actual);
    }
}
