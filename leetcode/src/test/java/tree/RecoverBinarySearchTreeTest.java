package tree;

import org.junit.Test;

import static tree.RecoverBinarySearchTree.*;

public class RecoverBinarySearchTreeTest {

    @Test
    public void test_example_1() {
        TreeNode n1 = new TreeNode(1);
        TreeNode n2 = new TreeNode(2);
        TreeNode n3 = new TreeNode(3);
        n1.left = n3;
        n3.right = n2;

        new RecoverBinarySearchTree().recoverTree(n1);
    }

    @Test
    public void test_wrong_1() {
        TreeNode n1 = new TreeNode(1);
        TreeNode n2 = new TreeNode(2);
        TreeNode n3 = new TreeNode(3);
        n2.left = n3;
        n2.right = n1;

        new RecoverBinarySearchTree().recoverTree(n2);
    }

    @Test
    public void test_my_1() {
        TreeNode n1 = new TreeNode(1);
        TreeNode n2 = new TreeNode(2);
        TreeNode n3 = new TreeNode(3);
        TreeNode n4 = new TreeNode(4);
        TreeNode n5 = new TreeNode(5);
        TreeNode n6 = new TreeNode(6);
        n4.left = n5;
        n4.right = n6;
        n5.left = n1;
        n5.right = n3;
        n6.left = n2;

        new RecoverBinarySearchTree().recoverTree(n4);
    }

    @Test
    public void test_wrong_2() {
        TreeNode n1 = new TreeNode(1);
        TreeNode n2 = new TreeNode(2);
        TreeNode n3 = new TreeNode(3);
        n3.right = n2;
        n2.right = n1;

        new RecoverBinarySearchTree().recoverTree(n3);
    }
}
