/**
 * LeetCode
 *
 * Problem 11: Container With Most Water
 */

package array;

public class ContainerWithMostWater {

    static public int maxArea(int[] height) {
        int l = 0;
        int r = height.length - 1;
        int max = -1;

        do {
            int area = (r - l) * Math.min(height[l], height[r]);
            max = Math.max(max, area);

            if (height[l] < height[r]) {
                l++;
            } else {
                r--;
            }

        } while (l < r);

        return max;
    }
}
