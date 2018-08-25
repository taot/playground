/**
 * LeetCode
 *
 * Problem 96: Unique Binary Search Trees
 */

package tree;

public class UniqueBinarySearchTrees {

    static public int numTrees(int n) {
        int[] N = new int[n+1];
        N[0] = 1;

        for (int i = 0; i <= n; i++) {
            for (int k = 0; k <= i-1; k++) {
                N[i] += N[k] * N[i-k-1];
            }
        }

        return N[n];
    }
}
