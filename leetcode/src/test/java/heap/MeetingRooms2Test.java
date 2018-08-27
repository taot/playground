package heap;

import org.junit.Assert;
import org.junit.Test;

import static heap.MeetingRooms2.*;

public class MeetingRooms2Test {

    @Test
    public void test_example_1() {
        Interval[] input = { new Interval(0,30), new Interval(5,10), new Interval(15,20) };
        int output = minMeetingRooms(input);
        Assert.assertEquals(2, output);
    }

    @Test
    public void test_example_2() {
        Interval[] input = { new Interval(7,10), new Interval(2,4) };
        int output = minMeetingRooms(input);
        Assert.assertEquals(1, output);
    }

    @Test
    public void test_my_2() {
        Interval[] input = { new Interval(1,10), new Interval(9,20), new Interval(5,20), new Interval(30,40) };
        int output = minMeetingRooms(input);
        Assert.assertEquals(3, output);
    }
    @Test
    public void test_wrong_1() {
        Interval[] input = { };
        int output = minMeetingRooms(input);
        Assert.assertEquals(0, output);
    }

}
