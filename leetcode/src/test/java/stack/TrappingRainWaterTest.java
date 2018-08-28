package stack;

import org.junit.Assert;
import org.junit.Test;

import static stack.TrappingRainWater.*;

public class TrappingRainWaterTest {

    @Test
    public void test_example_1() {
        int[] input = {0,1,0,2,1,0,1,3,2,1,2,1};
        int output = trap(input);
        Assert.assertEquals(6, output);
    }

    @Test
    public void test_my_1() {
        int[] input = {1,2,3,0,1,2,1,3,2};
        int output = trap(input);
        Assert.assertEquals(8, output);
    }

    @Test
    public void test_my_2() {
        int[] input = {};
        int output = trap(input);
        Assert.assertEquals(0, output);
    }

    @Test
    public void test_my_3() {
        int[] input = {5};
        int output = trap(input);
        Assert.assertEquals(0, output);
    }

    @Test
    public void test_my_4() {
        int[] input = {1,2,3,4,4,4,3,2,1};
        int output = trap(input);
        Assert.assertEquals(0, output);
    }

    @Test
    public void test_my_5() {
        int[] input = {5,0,0,0,4};
        int output = trap(input);
        Assert.assertEquals(12, output);
    }

    @Test
    public void test_wrong_1() {
        int[] input = {4,2,3};
        int output = trap(input);
        Assert.assertEquals(1, output);
    }
}
