/**
 * LeetCode
 *
 * Problem: 255: Verify Preorder Sequence in Binary Search Tree
 */

package tree;

public class VerifyPreorderSequenceInBinarySearchTree {

    static boolean verify(int[] A, int start, int end) {
        if (start >= end) {
            return true;
        }

        int x = A[start];
        int i = start + 1;
        while (i < end) {
            if (A[i] > x) {
                break;
            }
            i++;
        }

        while (i < end) {
            if (A[i] < x) {
                return false;
            }
            i++;
        }

        boolean flag = true;
        flag = flag && verify(A, start+1, i);
        flag = flag && verify(A, i, end);

        return flag;
    }

    static public boolean verifyPreorder(int[] preorder) {
        return verify(preorder, 0, preorder.length);
    }
}
