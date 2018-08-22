/**
 * LeetCode
 *
 * Problem 17: Letter Combinations of a Phone Number
 */

package string;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

public class LetterCombinationsPhoneNumber {

    static char[] letters2 = new char[] { 'a', 'b', 'c' };
    static char[] letters3 = new char[] { 'd', 'e', 'f' };
    static char[] letters4 = new char[] { 'g', 'h', 'i' };
    static char[] letters5 = new char[] { 'j', 'k', 'l' };
    static char[] letters6 = new char[] { 'm', 'n', 'o' };
    static char[] letters7 = new char[] { 'p', 'q', 'r', 's' };
    static char[] letters8 = new char[] { 't', 'u', 'v' };
    static char[] letters9 = new char[] { 'w', 'x', 'y', 'z' };
    static char[] letters_default = new char[] {};

    static public char[] getLetters(char digit) {
        switch (digit) {
            case '2':
                return letters2;
            case '3':
                return letters3;
            case '4':
                return letters4;
            case '5':
                return letters5;
            case '6':
                return letters6;
            case '7':
                return letters7;
            case '8':
                return letters8;
            case '9':
                return letters9;
            default:
                return letters_default;
        }
    }

    Set<String> results = new TreeSet<>();

    public void recurse(char[] cDigits, char[] buf, int n) {
        if (n == cDigits.length) {
            if (n == 0) {
                return;
            }
            String s = new String(buf);
            results.add(s);
            return;
        }
        char[] letters = getLetters(cDigits[n]);
        for (char c : letters) {
            buf[n] = c;
            recurse(cDigits, buf, n+1);
        }
    }

    public List<String> letterCombinations(String digits) {
        results.clear();
        char[] cDigits = digits.toCharArray();
        char[] buf = new char[cDigits.length];
        recurse(cDigits, buf, 0);
        return new ArrayList<>(results);
    }
}
