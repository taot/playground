/**
 * LeetCode
 *
 * Problem 239: Sliding Window Maximum
 */

package heap;

public class SlidingWindowMaximum {

    static public int[] maxSlidingWindow(int[] A, int k) {
        if (k <= 1) {
            return A;
        }
        if (A.length == 0) {
            return A;
        }

        int[] B = new int[A.length];
        int[] C = new int[A.length];
        int[] D = new int[A.length - k + 1];
        int nGroups = (A.length - 1) / k + 1;

        for (int g = 0; g < nGroups; g++) {
            // left max
            int max = Integer.MIN_VALUE;
            for (int i = 0; i < k; i++) {
                if (g * k + i >= A.length) {
                    continue;
                }
                if (A[g * k + i] > max) {
                    max = A[g * k + i];
                }
                B[g * k + i] = max;
            }
            // right max
            max = Integer.MIN_VALUE;
            for (int i = k-1; i >= 0; i--) {
                if (g * k + i >= A.length) {
                    continue;
                }
                if (A[g * k + i] > max) {
                    max = A[g * k + i];
                }
                C[g * k + i] = max;
            }
        }

        for (int i = 0; i < D.length; i++) {
            D[i] = Math.max(C[i], B[i + k - 1]);
        }

        return D;
    }
}
