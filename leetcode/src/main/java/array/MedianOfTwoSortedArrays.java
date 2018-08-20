/**
 * LeetCode
 *
 * Problem 4: Median of Two Sorted Arrays
 */

package array;

public class MedianOfTwoSortedArrays {

    static public double findMedianSortedArrays(int[] A, int[] B) {
        if (A.length > B.length) {
            int[] T = A;
            A = B;
            B = T;
        }
        int m = A.length;
        int n = B.length;

        int start = 0;
        int end = m;

        int i, j;

        while (true) {

            i = (start + end) / 2;
            j = (m + n + 1) / 2 - i;

            if ((i == 0 || A[i-1] <= B[j]) && (i == m || B[j-1] <= A[i])) {
                break;
            }

            if (i > 0 && A[i-1] > B[j]) {
                end = i;

            } else {
                start = i + 1;
            }
        }

        if ((m + n) % 2 == 0) {
            int left, right;
            if (i == 0) {
                left = B[j-1];
            } else if (j == 0) {
                left = A[i-1];
            } else {
                left = Math.max(A[i-1], B[j-1]);
            }
            if (j == n) {
                right = A[i];
            } else if (i == m) {
                right = B[j];
            } else {
                right = Math.min(A[i], B[j]);
            }
            return 0.5 * (left + right);

        } else {
            int res;
            if (i == 0) {
                res = B[j-1];
            } else if (j == 0) {
                res = A[i-1];
            } else {
                res = Math.max(A[i-1], B[j-1]);
            }
            return res;

        }
    }
}
