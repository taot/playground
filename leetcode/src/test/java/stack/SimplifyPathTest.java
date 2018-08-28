package stack;

import org.junit.Assert;
import org.junit.Test;

import static stack.SimplifyPath.*;

public class SimplifyPathTest {

    @Test
    public void test_example_1() {
        Assert.assertEquals("/home", simplifyPath("/home/"));
    }

    @Test
    public void test_example_2() {
        Assert.assertEquals("/c", simplifyPath("/a/./b/../../c/"));
    }

    @Test
    public void test_example_3() {
        Assert.assertEquals("/", simplifyPath("/../"));
    }

    @Test
    public void test_example_4() {
        Assert.assertEquals("/home/foo", simplifyPath("/home//foo/"));
    }

    @Test
    public void test_my_1() {
        Assert.assertEquals("/", simplifyPath(""));
    }
}
