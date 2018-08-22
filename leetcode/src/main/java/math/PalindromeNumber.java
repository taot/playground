/**
 * LeetCode
 *
 * Problem 9: Palindrome Number
 */

package math;

public class PalindromeNumber {

    static public boolean isPalindrome(int x) {
        int x0 = x;
        if (x < 0 || (x != 0 && x % 10 == 0)) {
            return false;
        }
        int n = 0;
        while (x > n) {
            n *= 10;
            n += x % 10;
            x /= 10;
        }
        return n == x || n / 10 == x;
    }
}
