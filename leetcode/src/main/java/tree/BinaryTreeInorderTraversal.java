/**
 * LeetCode
 *
 * Problem 94: Binary Tree Inorder Traversal
 */

package tree;

import java.util.ArrayList;
import java.util.List;

public class BinaryTreeInorderTraversal {

    static public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }

    static public void recurse(TreeNode node, List<Integer> result) {
        if (node == null) {
            return;
        }
        recurse(node.left, result);
        result.add(node.val);
        recurse(node.right, result);
    }

    static public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        recurse(root, result);
        return result;
    }
}
