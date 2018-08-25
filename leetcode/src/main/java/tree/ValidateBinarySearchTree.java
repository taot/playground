/**
 * LeetCode
 *
 * Problem 98: Validate Binary Search Tree
 */

package tree;

public class ValidateBinarySearchTree {

    static public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }

    static public class Result {
        boolean valid;
        int min;
        int max;

        public Result(boolean valid, int min, int max) {
            this.valid = valid;
            this.min = min;
            this.max = max;
        }
    }

    static public Result recurse(TreeNode root) {
        if (root == null) {
            return new Result(true, Integer.MAX_VALUE, Integer.MIN_VALUE);
        }

        Result rl = null;
        if (root.left != null) {
            rl = recurse(root.left);
            if (rl.max >= root.val || ! rl.valid) {
                return new Result(false, 0, 0);
            }
        }

        Result rr = null;
        if (root.right != null) {
            rr = recurse(root.right);
            if (rr.min <= root.val || ! rr.valid) {
                return new Result(false, 0, 0);
            }
        }

        int min = rl != null ? rl.min : root.val;
        int max = rr != null ? rr.max : root.val;

        return new Result(true, min, max);
    }

    static public boolean isValidBST(TreeNode root) {
        Result r = recurse(root);
        return r.valid;
    }

    static public boolean isValidBST2(TreeNode root) {
        if (root == null) {
            return true;
        }

        if (root.left != null && (root.val <= root.left.val || ! isValidBST(root.left))) {
            return false;
        }

        if (root.right != null && (root.val >= root.right.val || ! isValidBST(root.right))) {
            return false;
        }

        return true;
    }
}
