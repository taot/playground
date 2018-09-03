package graph;

import org.junit.Assert;
import org.junit.Test;

public class CourseScheduleTest {

    CourseSchedule obj = new CourseSchedule();

    @Test
    public void test_example_1() {
        int[][] prereq = {
            {1, 0}
        };
        int N = 2;
        boolean output = obj.canFinish(N, prereq);
        Assert.assertEquals(true, output);
    }

    @Test
    public void test_example_2() {
        int[][] prereq = {
                {1, 0},
                {0, 1}
        };
        int N = 2;
        boolean output = obj.canFinish(N, prereq);
        Assert.assertEquals(false, output);
    }

    @Test
    public void test_my_1() {
        int N = 6;
        int[][] prereq = {
                {0, 1},
                {1, 5},
                {0, 2},
                {2, 3},
                {2, 4}
        };

        boolean output = obj.canFinish(N, prereq);
        Assert.assertEquals(true, output);
    }

    @Test
    public void test_my_2() {
        int N = 6;
        int[][] prereq = {
                {0, 1},
                {1, 5},
                {0, 2},
                {2, 3},
                {2, 4},
                {4, 5}
        };

        boolean output = obj.canFinish(N, prereq);
        Assert.assertEquals(true, output);
    }

    @Test
    public void test_my_3() {
        int N = 6;
        int[][] prereq = {
                {0, 1},
                {1, 5},
                {0, 2},
                {2, 3},
                {2, 4},
                {4, 0}
        };

        boolean output = obj.canFinish(N, prereq);
        Assert.assertEquals(false, output);
    }
}
