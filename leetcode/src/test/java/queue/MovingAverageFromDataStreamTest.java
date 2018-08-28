package queue;

import org.junit.Assert;
import org.junit.Test;
import queue.MovingAverage;

public class MovingAverageFromDataStreamTest {

    static final double EPS = 0.001;

    @Test
    public void test_example_1() {
        MovingAverage m = new MovingAverage(3);
        Assert.assertEquals(1, m.next(1), EPS);
        Assert.assertEquals((1 + 10) / 2.0, m.next(10), EPS);
        Assert.assertEquals((1 + 10 + 3) / 3.0, m.next(3), EPS);
        Assert.assertEquals((10 + 3 + 5) / 3.0, m.next(5), EPS);
    }
}
