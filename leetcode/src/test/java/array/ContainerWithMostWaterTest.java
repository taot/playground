package array;

import org.junit.Assert;
import org.junit.Test;

import static array.ContainerWithMostWater.*;

public class ContainerWithMostWaterTest {

    @Test
    public void test_example_1() {
        int v = maxArea(new int[] { 1,8,6,2,5,4,8,3,7 });
        Assert.assertEquals(49, v);
    }

    @Test
    public void test_my_1() {
        int v = maxArea(new int[] { 1, 2 });
        Assert.assertEquals(1, v);
    }

    @Test
    public void test_my_2() {
        int v = maxArea(new int[] { 1, 5, 4 });
        Assert.assertEquals(4, v);
    }

    @Test
    public void test_my_3() {
        int v = maxArea(new int[] { 1, 2, 3, 4, 5 });
        Assert.assertEquals(6, v);
    }
}
