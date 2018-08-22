/**
 * LeetCode
 *
 * Problem 18: 4Sum
 */

package array;

import java.util.*;

public class FourSum {

    static public List<List<Integer>> fourSum(int[] nums, int target) {
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

        for (int i = 0; i < nums.length - 1; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                int k = 0;
                int l = nums.length - 1;
                while (k < l) {
                    if (k == i || k == j || (k > 0 && nums[k] == nums[k-1])) {
                        k++;
                        continue;
                    }
                    if (l == i || l == j || (l < nums.length - 1 && nums[l] == nums[l+1])) {
                        l--;
                        continue;
                    }

                    int sum = nums[i] + nums[j] + nums[k] + nums[l];
                    if (sum == target) {
                        List<Integer> list = new ArrayList<>();
                        list.add(nums[i]);
                        list.add(nums[j]);
                        list.add(nums[k]);
                        list.add(nums[l]);
                        Collections.sort(list);
                        set.add(list);

                        k++;

                    } else if (sum < target) {
                        k++;
                    } else {
                        l--;
                    }
                }
            }
        }

        return new ArrayList(set);
    }
}
