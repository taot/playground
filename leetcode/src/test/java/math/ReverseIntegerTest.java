package math;

import org.junit.Assert;
import org.junit.Test;

import static math.ReverseInteger.*;

public class ReverseIntegerTest {

    @Test
    public void test_example_1() {
        int x = reverse(123);
        Assert.assertEquals(321, x);
    }

    @Test
    public void test_example_2() {
        int x = reverse(-123);
        Assert.assertEquals(-321, x);
    }

    @Test
    public void test_example_3() {
        int x = reverse(120);
        Assert.assertEquals(21, x);
    }

    @Test
    public void test_my_1() {
        int x = reverse(0);
        Assert.assertEquals(0, x);
    }

    @Test
    public void test_wrong_1() {
        int x = reverse(1534236469);
        Assert.assertEquals(0, x);
    }

}
