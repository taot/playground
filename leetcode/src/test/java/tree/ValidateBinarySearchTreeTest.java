package tree;

import org.junit.Assert;
import org.junit.Test;

import static tree.ValidateBinarySearchTree.*;

public class ValidateBinarySearchTreeTest {

    @Test
    public void test_example_1() {
        TreeNode n2 = new TreeNode(2);
        TreeNode n1 = new TreeNode(1);
        TreeNode n3 = new TreeNode(3);
        n2.left = n1;
        n2.right = n3;

        Assert.assertTrue(isValidBST(n2));
    }

    @Test
    public void test_example_2() {
        TreeNode n1 = new TreeNode(1);
        TreeNode n3 = new TreeNode(3);
        TreeNode n4 = new TreeNode(4);
        TreeNode n5 = new TreeNode(5);
        TreeNode n6 = new TreeNode(6);

        n5.left = n1;
        n5.right = n4;
        n4.left = n3;
        n4.right = n6;

        Assert.assertFalse(isValidBST(n5));
    }

    @Test
    public void test_wrong_1() {
        TreeNode n1 = new TreeNode(1);
        TreeNode n2 = new TreeNode(1);
        n2.left = n1;

        Assert.assertFalse(isValidBST(n2));
    }

    @Test
    public void test_wrong_2() {
        TreeNode n5 = new TreeNode(5);
        TreeNode n6 = new TreeNode(6);
        TreeNode n10 = new TreeNode(10);
        TreeNode n15 = new TreeNode(15);
        TreeNode n20 = new TreeNode(20);

        n10.left = n5;
        n10.right = n15;
        n15.left = n6;
        n15.right = n20;

        Assert.assertFalse(isValidBST(n10));
    }
}
