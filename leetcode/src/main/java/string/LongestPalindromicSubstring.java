/**
 * LeetCode
 *
 * Problem 5. Longest Palindromic Substring
 */

package string;

public class LongestPalindromicSubstring {

    static public String longestPalindrome(String s) {
        int len = s.length();

        int max = 0;
        int start = 0;
        int end = -1;
        for (int i = 0; i < len; i++) {
            for (int k = 0; i - k >= 0 && i + k < len; k++) {
                if (s.charAt(i-k) != s.charAt(i+k)) {
                    break;
                }
                if (k * 2 + 1 > max) {
                    max = k * 2 + 1;
                    start = i-k;
                    end = i+k;
                }
            }
            for (int k = 0; i - k >= 0 && i + k + 1 < len; k++) {
                if (s.charAt(i-k) != s.charAt(i+k+1)) {
                    break;
                }
                if (2 * k + 2 > max) {
                    max = 2 * k + 2;
                    start = i - k;
                    end = i + k + 1;
                }
            }
        }
        return s.substring(start, end + 1);
    }
}
