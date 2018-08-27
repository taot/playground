package stack;

import org.junit.Assert;
import org.junit.Test;

import static stack.ValidParentheses.*;

public class ValidParenthesesTest {

    @Test
    public void test_example_1() {
        Assert.assertTrue(isValid("()"));
    }

    @Test
    public void test_example_2() {
        Assert.assertTrue(isValid("()[]{}"));
    }

    @Test
    public void test_example_3() {
        Assert.assertFalse(isValid("(]"));
    }

    @Test
    public void test_example_4() {
        Assert.assertFalse(isValid("([)]"));
    }

    @Test
    public void test_example_5() {
        Assert.assertTrue(isValid("{[]}"));
    }
}
