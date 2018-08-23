package tree;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static tree.BinaryTreeInorderTraversal.*;

public class BinaryTreeInorderTraversalTest {

    @Test
    public void test_example_1() {
        TreeNode root = new TreeNode(1);
        TreeNode node2 = new TreeNode(2);
        TreeNode node3 = new TreeNode(3);
        root.right = node2;
        node2.left = node3;

        List result = inorderTraversal(root);
        Assert.assertEquals(Arrays.asList(1, 3, 2), result);
    }
}
