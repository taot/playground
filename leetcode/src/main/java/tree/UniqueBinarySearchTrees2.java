/**
 * LeetCode
 *
 * Problem 95: Unique Binary Search Trees II
 */

package tree;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class UniqueBinarySearchTrees2 {

    static public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }

    static public List<TreeNode> gen(int start, int end) {
        if (start == end) {
            return Collections.emptyList();
        }

        List<TreeNode> nodes = new ArrayList<>();

        for (int i = start; i < end; i++) {
            List<TreeNode> lefts = gen(start, i);
            List<TreeNode> rights = gen(i+1, end);
            int root = i+1;


            if (lefts.isEmpty() && rights.isEmpty()) {
                TreeNode n = new TreeNode(root);
                n.left = n.right = null;
                nodes.add(n);

            } else if (lefts.isEmpty()) {
                for (TreeNode rn : rights) {
                    TreeNode n = new TreeNode(root);
                    n.left = null;
                    n.right = rn;
                    nodes.add(n);
                }

            } else if (rights.isEmpty()) {
                for (TreeNode ln : lefts) {
                    TreeNode n = new TreeNode(root);
                    n.left = ln;
                    n.right = null;
                    nodes.add(n);
                }

            } else {
                for (TreeNode ln : lefts) {
                    for (TreeNode rn : rights) {
                        TreeNode n = new TreeNode(root);
                        n.left = ln;
                        n.right = rn;
                        nodes.add(n);
                    }
                }

            }
        }

        return nodes;
    }

    static public List<TreeNode> generateTrees(int n) {
        return gen(0, n);
    }

    static public List<TreeNode> generateTrees2(int n) {
        List<TreeNode>[][] TS = new List[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                TS[i][j] = new ArrayList<>();
            }
        }

        for (int l = 1; l <= n; l++) {
            for (int s = 0; s < n-l; s++) {
                for (int r = 0; r <= l-1; r++) {
                    List<TreeNode> lefts = TS[s][s+r];
                    List<TreeNode> rights = TS[s+r+1][s+l];
                    int root = s+r+1;

                    if (lefts.isEmpty() && rights.isEmpty()) {
                        TreeNode newNode = new TreeNode(root);
                        newNode.left = newNode.right = null;
                        TS[s][s+l].add(newNode);

                    } else if (lefts.isEmpty()) {
                        for (TreeNode rn : rights) {
                            TreeNode newNode = new TreeNode(root);
                            newNode.left = null;
                            newNode.right = rn;
                            TS[s][s+l].add(newNode);
                        }

                    } else if (rights.isEmpty()) {
                        for (TreeNode ln: lefts) {
                            TreeNode newNode = new TreeNode(root);
                            newNode.left = ln;
                            newNode.right = null;
                            TS[s][s+l].add(newNode);
                        }

                    } else {
                        for (TreeNode ln : lefts) {
                            for (TreeNode rn : rights) {
                                TreeNode newNode = new TreeNode(root);
                                newNode.left = ln;
                                newNode.right = rn;
                                TS[s][s + l].add(newNode);
                            }
                        }
                    }
                }
            }
        }

        return TS[0][n-1];
    }
}
