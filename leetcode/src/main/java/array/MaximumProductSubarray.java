/**
 * LeetCode
 *
 * 152. Maximum Product Subarray
 */

package array;

public class MaximumProductSubarray {

    public int maxProduct(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }

        int[] P = new int[nums.length];
        int[] N = new int[nums.length];

        if (nums[0] >= 0) {
            P[0] = nums[0];
            N[0] = 0;
        } else {
            P[0] = 0;
            N[0] = nums[0];
        }

        for (int i = 1; i < nums.length; i++) {
            if (nums[i] >= 0) {
                P[i] = P[i-1] == 0 ? nums[i] : P[i-1] * nums[i];
                N[i] = N[i-1] == 0 ? 0 : N[i-1] * nums[i];
            } else {
                P[i] = N[i-1] == 0 ? 0 : N[i-1] * nums[i];
                N[i] = P[i-1] == 0 ? nums[i] : P[i-1] * nums[i];
            }
        }

        int max = -1;
        for (int i = 0; i < nums.length; i++) {
            if (P[i] > max) {
                max = P[i];
            }
        }

        return max;
    }
}
