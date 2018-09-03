package array;

import org.junit.Assert;
import org.junit.Test;

public class MaximumProductSubarrayTest {

    MaximumProductSubarray algo = new MaximumProductSubarray();

    @Test
    public void test_example_1() {
        int[] input = {2,3,-2,4};
        int output = algo.maxProduct(input);
        Assert.assertEquals(6, output);
    }

    @Test
    public void test_example_2() {
        int[] input = {-1,0,-1};
        int output = algo.maxProduct(input);
        Assert.assertEquals(0, output);
    }

    @Test
    public void test_my_1() {
        int[] input = {2,3,-2,4,-1};
        int output = algo.maxProduct(input);
        Assert.assertEquals(48, output);
    }

    @Test
    public void test_my_2() {
        int[] input = {0,5,0,-2,-3,0,4};
        int output = algo.maxProduct(input);
        Assert.assertEquals(6, output);
    }

    @Test
    public void test_wrong_1() {
        int[] input = {-2};
        int output = algo.maxProduct(input);
        Assert.assertEquals(-2, output);
    }

    @Test
    public void test_wrong_2() {
        int[] input = {0,2};
        int output = algo.maxProduct(input);
        Assert.assertEquals(2, output);
    }
}
