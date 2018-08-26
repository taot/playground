package heap;

import org.junit.Assert;
import org.junit.Test;
import static heap.KthLargestElementInAnArray.*;

public class KthLargestElementInAnArrayTest {

    @Test
    public void test_example_1() {
        int[] a = { 3,2,1,5,6,4 };
        Assert.assertEquals(5, findKthLargest(a, 2));
    }

    @Test
    public void test_example_2() {
        int[] a = { 3,2,3,1,2,4,5,5,6 };
        Assert.assertEquals(4, findKthLargest(a, 4));
    }

    @Test
    public void test_my_1() {
        int[] a = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14 };
        Assert.assertEquals(11, findKthLargest(a, 4));
    }
}
