package array;

import org.junit.Assert;
import org.junit.Test;

public class TwoSumTest {

    @Test
    public void test_basic() {
        int[] nums = { 2, 7, 11, 15 };
        int target = 9;
        int[] sln = TwoSum.twoSum(nums, target);
        Assert.assertArrayEquals(new int[] { 0, 1 }, sln);
    }
}
