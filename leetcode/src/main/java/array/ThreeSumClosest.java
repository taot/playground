/**
 * LeetCode
 *
 * Problem 16: 3Sum Closest
 */

package array;

import java.util.Arrays;

public class ThreeSumClosest {

    static public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);

        int diffMin = Integer.MAX_VALUE;
        int sum = 0;

        for (int i = 0; i < nums.length; i++) {
            int j = 0;
            int k = nums.length - 1;

            while (j < k) {
                if (j == i || (j > 0 && j != i && nums[j] == nums[j-1])) {
                    j++;
                    continue;
                }
                if (k == i || (k < nums.length - 1 && k != i && nums[k] == nums[k+1])) {
                    k--;
                    continue;
                }

                int diff = nums[j] + nums[k] + nums[i] - target;
                if (Math.abs(diff) < diffMin) {
                    diffMin = Math.abs(diff);
                    sum = nums[j] + nums[k] + nums[i];
                }

                if (diff == 0) {
                    return target;
                } else if (diff < 0) {
                    j++;
                } else {
                    k--;
                }
            }
        }

        return sum;
    }
}
