/**
 * LeetCode
 *
 * Problem 346: Moving Average from Data Stream
 */

package queue;

import java.util.ArrayDeque;
import java.util.Queue;

class MovingAverage {

    private int size;
    private Queue<Integer> queue = new ArrayDeque<>();
    private int sum = 0;

    public MovingAverage(int size) {
        this.size = size;
    }

    public double next(int val) {
        queue.offer(val);
        sum += val;
        if (queue.size() > size) {
            int n = queue.poll();
            sum -= n;
        }
        return sum * 1.0 / queue.size();
    }
}

public class MovingAverageFromDataStream {
}
