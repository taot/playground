/**
 * LeetCode
 *
 * Problem 253: Meeting Rooms II
 */

package heap;

import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;

public class MeetingRooms2 {

    static public class Interval {
        int start;
        int end;
        Interval() { start = 0; end = 0; }
        Interval(int s, int e) { start = s; end = e; }
    }

    static public int minMeetingRooms(Interval[] intervals) {
        int[] S = new int[intervals.length];
        int[] E = new int[intervals.length];
        for (int i = 0; i < intervals.length; i++) {
            S[i] = intervals[i].start;
            E[i] = intervals[i].end;
        }
        Arrays.sort(S);
        Arrays.sort(E);

        int i = 0;
        int j = 0;
        int max = 0;
        int count = 0;

        while (j < intervals.length) {
            if (i < intervals.length && S[i] < E[j]) {
                count++;
                if (count > max) {
                    max = count;
                }
                i++;
            } else {
                j++;
                count--;
            }
        }

        return max;
    }

    static public int minMeetingRooms2(Interval[] intervals) {
        if (intervals == null || intervals.length == 0) {
            return 0;
        }

        Arrays.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.start - o2.start;
            }
        });

        PriorityQueue<Interval> heap = new PriorityQueue<>(new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.end - o2.end;
            }
        });

        heap.add(intervals[0]);
        int i = 1;
        int minSize = 1;

        while (i < intervals.length && ! heap.isEmpty()) {
            Interval a = heap.peek();
            if (intervals[i].start < a.end) {
                heap.add(intervals[i]);
                if (heap.size() > minSize) {
                    minSize = heap.size();
                }
            } else {
                heap.poll();
                heap.add(intervals[i]);
            }
            i++;
        }

        return minSize;
    }
}
