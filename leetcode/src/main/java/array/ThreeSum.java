/**
 * LeetCode
 *
 * Problem 15: 3Sum
 */

package array;

import java.util.*;

public class ThreeSum {

    static public List<List<Integer>> threeSum(int[] nums) {
        List<Integer> numsList = new ArrayList<>();
        int zeroCount = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                if (zeroCount == 0) {
                    numsList.add(0);
                }
                zeroCount++;
            } else {
                numsList.add(nums[i]);
            }
        }

        nums = new int[numsList.size()];
        for (int i = 0; i < numsList.size(); i++) {
            nums[i] = numsList.get(i);
        }

        Arrays.sort(nums);

        Set<List<Integer>> set = new TreeSet<>(new Comparator<List<Integer>>() {
            @Override
            public int compare(List<Integer> o1, List<Integer> o2) {
                if (o1.size() < o2.size()) {
                    return -1;
                }
                if (o1.size() > o2.size()) {
                    return 1;
                }
                for (int i = 0; i < o1.size(); i++) {
                    if (o1.get(i) < o2.get(i)) {
                        return -1;
                    }
                    if (o1.get(i) > o2.get(i)) {
                        return 1;
                    }
                }
                return 0;
            }
        });

        for (int i = 0; i < nums.length; i++) {

            int expectedSum = -nums[i];

            int j = 0;
            int k = nums.length - 1;
            while (j < k) {
                if (j == i) {
                    j++;
                    continue;
                }
                if (k == i) {
                    k--;
                    continue;
                }

                int sum = nums[j] + nums[k];
                if (sum == expectedSum) {
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[i]);
                    list.add(nums[j]);
                    list.add(nums[k]);
                    Collections.sort(list);
                    set.add(list);

                    j++;

                } else if (sum < expectedSum) {
                    j++;
                } else {
                    k--;
                }
            }
        }

        if (zeroCount > 2) {
            List<Integer> list = new ArrayList<>();
            list.add(0);
            list.add(0);
            list.add(0);
            set.add(list);
        }

        return new ArrayList(set);
    }
}
