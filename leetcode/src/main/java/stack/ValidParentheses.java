/**
 * LeetCode
 *
 * Problem 20: Valid Parentheses
 */

package stack;

import java.util.Stack;

public class ValidParentheses {

    static public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '[' || c == '{') {
                stack.push(c);
            } else {
                if (stack.isEmpty()) {
                    return false;
                }
                char d = stack.pop();
                if (d == '(' && c != ')') {
                    return false;
                }
                if (d == '[' && c != ']') {
                    return false;
                }
                if (d == '{' && c != '}') {
                    return false;
                }
            }
        }

        return stack.isEmpty();
    }
}
