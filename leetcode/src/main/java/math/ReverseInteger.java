/**
 * LeetCode
 *
 * Problem 7: Reverse Integer
 */

package math;

public class ReverseInteger {

    static public int reverse(int x) {
        int MAX = 2147483647;
        boolean negative = false;
        if (x < 0) {
            negative = true;
            x = -x;
        }
        int n = 0;
        while (x > 0) {
            if (n > MAX / 10) {
                return 0;
            }
            n *= 10;
            n += x % 10;
            if (n < 0) {
                return 0;
            }
            x /= 10;
        }
        if (negative) {
            n = -n;
        }
        return n;
    }
}
