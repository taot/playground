package math;

import org.junit.Assert;
import org.junit.Test;

import static math.PalindromeNumber.*;

public class PalindromeNumberTest {

    @Test
    public void test_example_1() {
        Assert.assertTrue(isPalindrome(121));
    }

    @Test
    public void test_example_2() {
        Assert.assertFalse(isPalindrome(-121));
    }

    @Test
    public void test_example_3() {
        Assert.assertFalse(isPalindrome(10));
    }
}
