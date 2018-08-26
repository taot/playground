/**
 * LeetCode
 *
 * Problem 215: Kth Largest Element in an Array
 */

package heap;

import java.util.Comparator;
import java.util.PriorityQueue;

public class KthLargestElementInAnArray {

    /*
     * 闭开区间 [start, end)
     */
    static int split(int[] A, int start, int end) {
        if (start == end) {
            return start;
        }
        int x = A[end - 1];
        int i = start;
        for (int j = start; j <= end - 2; j++) {
            if (A[j] > x) {
                swap(A, i, j);
                i++;
            }
        }
        swap(A, i, end - 1);

        return i;
    }

    static void swap(int[] A, int i, int j) {
        int tmp = A[i];
        A[i] = A[j];
        A[j] = tmp;
    }

    static int recurse(int[] A, int start, int end, int k) {
        int i = split(A, start, end);
        if (i == k) {
            return A[i];
        }

        if (i < k) {
            return recurse(A, i+1, end, k);
        } else {
            return recurse(A, start, i, k);
        }
    }

    static public int findKthLargest(int[] nums, int k) {
        return recurse(nums, 0, nums.length, k-1);
    }

    static public int findKthLargest1(int[] nums, int k) {
        PriorityQueue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });
        for (int n : nums) {
            queue.add(n);
        }

        int n = 0;
        for (int i = 0; i < k; i++) {
            n = queue.poll();
        }

        return n;
    }
}
