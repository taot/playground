/**
 * LeetCode
 *
 * Problem 71: Simplify Path
 */

package stack;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class SimplifyPath {

    static public String simplifyPath(String path) {
        List<String> tokens = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < path.length(); i++) {
            char c = path.charAt(i);
            if (c == '/') {
                if (sb.length() > 0) {
                    tokens.add(sb.toString());
                    sb.delete(0, sb.length());
                }
            } else {
                sb.append(c);
            }
        }
        if (sb.length() > 0) {
            tokens.add(sb.toString());
        }

        Stack<String> stack = new Stack<>();
        for (String t : tokens) {
            if (".".equals(t)) {
                // do nothing
            } else if ("..".equals(t)) {
                if (! stack.isEmpty()) {
                    stack.pop();
                }
            } else {
                stack.push(t);
            }
        }

        sb = new StringBuilder();
        for (String s : stack.toArray(new String[stack.size()])) {
            sb.append('/');
            sb.append(s);
        }
        if (sb.length() == 0) {
            sb.append('/');
        }

        return sb.toString();
    }
}
