package tree;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static tree.BinaryTreeLevelOrderTraversal.*;

public class BinaryTreeLevelOrderTraversalTest {

    @Test
    public void test_example_1() {
        TreeNode n3 = new TreeNode(3);
        TreeNode n7 = new TreeNode(7);
        TreeNode n9 = new TreeNode(9);
        TreeNode n15 = new TreeNode(15);
        TreeNode n20 = new TreeNode(20);

        n3.left = n9;
        n3.right = n20;
        n20.left = n15;
        n20.right = n7;

        List<List<Integer>> actual = new BinaryTreeLevelOrderTraversal().levelOrder(n3);
        List<List<Integer>> expected = Arrays.asList(
                Arrays.asList(3),
                Arrays.asList(9, 20),
                Arrays.asList(15, 7)
        );

        Assert.assertEquals(expected, actual);
    }

    @Test
    public void test_wrong_1() {
        List<List<Integer>> actual = new BinaryTreeLevelOrderTraversal().levelOrder(null);
        List<List<Integer>> expected = Collections.emptyList();

        Assert.assertEquals(expected, actual);
    }
}
