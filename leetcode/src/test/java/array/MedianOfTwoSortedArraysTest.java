package array;

import org.junit.Assert;
import org.junit.Test;
import static array.MedianOfTwoSortedArrays.*;

public class MedianOfTwoSortedArraysTest {

    @Test
    public void test_example_1() {
        double r = findMedianSortedArrays(new int[] {1, 3}, new int[] {2});
        Assert.assertEquals(2, r, 1E-3);
    }

    @Test
    public void test_example_2() {
        double r = findMedianSortedArrays(new int[] {1, 2}, new int[] {3, 4});
        Assert.assertEquals(2.5, r, 1E-3);
    }

    @Test
    public void test_my_data_1() {
        double r = findMedianSortedArrays(new int[] {1, 2, 3, 4}, new int[] {5, 6, 7, 8});
        Assert.assertEquals(4.5, r, 1E-3);
    }

    @Test
    public void test_my_data_2() {
        double r = findMedianSortedArrays(new int[] {}, new int[] {5, 6, 7, 8});
        Assert.assertEquals(6.5, r, 1E-3);
    }

    @Test
    public void test_my_data_3() {
        double r = findMedianSortedArrays(new int[] {5, 6, 7, 10}, new int[] {1, 2, 9});
        Assert.assertEquals(6, r, 1E-3);
    }

    @Test
    public void test_my_data_4() {
        double r = findMedianSortedArrays(new int[] {1, 3, 5, 7}, new int[] {2, 4, 6, 8});
        Assert.assertEquals(4.5, r, 1E-3);
    }

    @Test
    public void test_my_data_5() {
        double r = findMedianSortedArrays(new int[] {3, 5, 7}, new int[] {2, 4, 6, 8});
        Assert.assertEquals(5, r, 1E-3);
    }
}
