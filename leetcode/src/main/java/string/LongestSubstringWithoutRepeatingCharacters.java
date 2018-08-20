/**
 * LeetCode
 *
 * Problem 3: Longest Substring Without Repeating Characters
 */

package string;

import java.util.HashMap;
import java.util.HashSet;

public class LongestSubstringWithoutRepeatingCharacters {

    /* Sliding window */
    static public int lengthOfLongestSubstring(String s) {
        int max = 0;
        int i = 0;
        int j = 0;
        HashSet<Character> set = new HashSet<>();

        while (i < s.length() && j < s.length()) {
            char c = s.charAt(j);
            if (set.contains(c)) {
                set.remove(s.charAt(i));
                i++;
            } else {
                set.add(c);
                j++;
                if (j - i > max) {
                    max = j - i;
                }
            }
        }

        return max;
    }

    /* Not beautiful */
    static public int lengthOfLongestSubstring2(String s) {
        char[] chars = s.toCharArray();
        int len = 0;
        int start = 0;

        int max = 0;
        HashMap<Character, Integer> counts = new HashMap<>();
        boolean dup = false;
        while (start <= chars.length - 1) {
            if (dup || start + len >= chars.length) {
                char c = chars[start];
                if (counts.get(c) != null) {
                    int cnt = counts.get(c);
                    if (cnt >= 2) {
                        dup = false;
                        counts.put(c, cnt - 1);
                    } else {
                        counts.remove(c);
                    }
                }

                start++;
                len--;

            } else {
                len++;
                char c = chars[start + len - 1];
                Integer cnt = counts.get(c);
                if (cnt == null) {
                    counts.put(c, 1);
                    if (len > max) {
                        max = len;
                    }
                } else {
                    dup = true;
                    counts.put(c , cnt + 1);
                }
            }

        }

        return max;
    }

    /* Timeout */
    static public int lengthOfLongestSubstring1(String s) {
        char[] chars = s.toCharArray();
        int max = 0;
        for (int i = 0; i < chars.length; i++) {
            for (int j = i; j < chars.length; j++) {
                HashMap<Character, Boolean> map = new HashMap<>();
                boolean flag = true;
                for (int k = i; k <= j; k++) {
                    char c = chars[k];
                    if (map.containsKey(c)) {
                        flag = false;
                        break;
                    }
                    map.put(c, true);
                }
                if (flag) {
                    int l = j - i + 1;
                    if (l > max) {
                        max = l;
                    }
                }
            }
        }
        return max;
    }
}
