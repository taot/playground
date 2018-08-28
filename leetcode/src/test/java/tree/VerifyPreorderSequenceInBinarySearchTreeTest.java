package tree;

import org.junit.Assert;
import org.junit.Test;

import static tree.VerifyPreorderSequenceInBinarySearchTree.*;

public class VerifyPreorderSequenceInBinarySearchTreeTest {

    @Test
    public void test_example_1() {
        int[] input = {5,2,6,1,3};
        Assert.assertFalse(verifyPreorder(input));
    }

    @Test
    public void test_example_2() {
        int[] input = {5,2,1,3,6};
        Assert.assertTrue(verifyPreorder(input));
    }
}
