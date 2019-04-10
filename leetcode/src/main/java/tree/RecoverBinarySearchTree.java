/**
 * LeetCode
 *
 * Problem 99: Recover Binary Search Tree
 */

package tree;

public class RecoverBinarySearchTree {

    static public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }

        @Override
        public String toString() {
            return "TreeNode{" +
                    "val=" + val +
                    '}';
        }
    }

    TreeNode node1;
    TreeNode node2;
    TreeNode node3;

    void recurse(TreeNode root, int min, int max, TreeNode minNode, TreeNode maxNode) {
        if (root == null) {
            return;
        }

        if (root.val > min) {
            node1 = minNode;
            if (node2 == null) {
                node2 = root;
            } else {
                node3 = root;
            }
//            return;
        }

        if (root.val < max) {
            node1 = maxNode;
            if (node2 == null) {
                node2 = root;
            } else {
                node3 = root;
            }
//            return;
        }

        recurse(root.left, root.val, max, root, maxNode);
        recurse(root.right, min, root.val, minNode, root);
    }

    public void recoverTree(TreeNode root) {
        recurse(root, Integer.MAX_VALUE, Integer.MIN_VALUE, root, root);
        System.out.println(node1);
        System.out.println(node2);
        System.out.println(node3);
        if (node2 == null) {
            int tmp = node1.val;
            node1.val = node3.val;
            node3.val = tmp;
        } else if (node3 == null) {
            int tmp = node1.val;
            node1.val = node2.val;
            node2.val = tmp;
        } else {
            int tmp = node2.val;
            node2.val = node3.val;
            node3.val = tmp;
        }

    }
}
