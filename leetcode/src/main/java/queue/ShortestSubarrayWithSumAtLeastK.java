/**
 * LeetCode
 *
 * Problem 862: Shortest Subarray with Sum at Least K
 */

package queue;

import java.util.Deque;
import java.util.LinkedList;

public class ShortestSubarrayWithSumAtLeastK {

    static public int shortestSubarray(int[] A, int K) {
        int N = A.length;
        long[] P = new long[N + 1];
        P[0] = 0;
        for (int i = 0; i < N; i++) {
            P[i+1] = P[i] + A[i];
        }

        long min = N+1;
        Deque<Integer> q = new LinkedList<>();
        for (int i = 0; i < P.length; i++) {
            while (! q.isEmpty() && P[q.getLast()] > P[i]) {
                q.removeLast();
            }
            while (! q.isEmpty() && P[i] - P[q.getFirst()] >= K) {
                min = Math.min(i - q.getFirst(), min);
                q.removeFirst();
            }
            q.add(i);
        }

        return min == N+1 ? -1 : (int) min;
    }
}
