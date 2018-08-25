/**
 * LeetCode
 *
 * Problem 102: Binary Tree Level Order Traversal
 */

package tree;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class BinaryTreeLevelOrderTraversal {

    static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }

    static class Item {
        TreeNode node;
        int level;

        public Item(TreeNode node, int level) {
            this.node = node;
            this.level = level;
        }
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> lists = new ArrayList<>();

        if (root == null) {
            return lists;
        }

        Queue<Item> queue = new LinkedList<>();

        queue.add(new Item(root, 0));
        int level = -1;

        List<Integer> lastList = null;

        while (! queue.isEmpty()) {
            Item it = queue.poll();
            if (it.level > level) {
                level = it.level;
                lastList = new ArrayList<>();
                lists.add(lastList);
            }
            TreeNode n = it.node;
            lastList.add(n.val);
            if (n.left != null) {
                queue.add(new Item(n.left, level + 1));
            }
            if (n.right != null) {
                queue.add(new Item(n.right, level+1));
            }
        }

        return lists;
    }
}
