package string;

import org.junit.Assert;
import org.junit.Test;

import static string.ZigZagConversion.*;

public class ZigZagConversionTest {

    @Test
    public void test_example_1() {
        String s = convert("PAYPALISHIRING", 3);
        Assert.assertEquals("PAHNAPLSIIGYIR", s);
    }

    @Test
    public void test_example_2() {
        String s = convert("PAYPALISHIRING", 4);
        Assert.assertEquals("PINALSIGYAHRPI", s);
    }

    @Test
    public void test_my_1() {
        String s = convert("ABCDEFG", 2);
        Assert.assertEquals("ACEGBDF", s);
    }

    @Test
    public void test_my_2() {
        String s = convert("ABCDEFG", 1);
        Assert.assertEquals("ABCDEFG", s);
    }
}
