/**
 * LeetCode
 *
 * Problem 42: Trapping Rain Water
 */

package stack;

import java.util.Stack;

public class TrappingRainWater {

    static public int trap(int[] height) {
        int total = 0;
        Stack<Integer> stack = new Stack<>();

        for (int i = 0; i < height.length; i++) {
            while (! stack.isEmpty() && height[stack.peek()] < height[i]) {
                int top = stack.pop();
                if (! stack.isEmpty()) {
                    int min = Math.min(height[stack.peek()], height[i]);
                    total += (min - height[top]) * (i - stack.peek() - 1);
                }

            }
            stack.push(i);
        }

        return total;
    }
}
