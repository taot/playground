/**
 * LeetCode
 *
 * Problem 454: 4Sum II
 */

package hashtable;

import java.util.HashMap;
import java.util.Map;

public class FourSum {

    static public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        if (A.length == 0 || B.length == 0 || C.length == 0 || D.length == 0) {
            return 0;
        }

        Map<Integer, Integer> map = new HashMap<>();
        for (int c : C) {
            for (int d : D) {
                Integer n = map.get(c+d);
                if (n == null) {
                    n = 0;
                }
                n++;
                map.put(c+d, n);
            }
        }

        int count = 0;

        for (int a : A) {
            for (int b : B) {
                int s = a + b;
                Integer k = map.get(-s);
                if (k != null) {
                    count += k;
                }
            }
        }

        return count;
    }

    static public int fourSumCount2(int[] A, int[] B, int[] C, int[] D) {
        if (A.length == 0 || B.length == 0 || C.length == 0 || D.length == 0) {
            return 0;
        }

        Map<Integer, Integer> map = new HashMap<>();
        for (int d : D) {
            Integer n = map.get(d);
            if (n == null) {
                n = 0;
            }
            n++;
            map.put(d, n);
        }

        int count = 0;

        for (int a : A) {
            for (int b : B) {
                for (int c : C) {
                    int s = a + b + c;
                    Integer k = map.get(-s);
                    if (k != null) {
                        count += k;
                    }
                }
            }
        }

        return count;
    }
}
