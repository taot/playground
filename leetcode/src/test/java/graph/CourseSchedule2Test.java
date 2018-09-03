package graph;

import org.junit.Assert;
import org.junit.Test;

public class CourseSchedule2Test {

    CourseSchedule2 obj = new CourseSchedule2();

    @Test
    public void test_example_1() {
        int[][] prereq = {
            {1, 0}
        };
        int N = 2;
        int[] output = obj.findOrder(N, prereq);
        Assert.assertArrayEquals(new int[] {0,1}, output);
    }

    @Test
    public void test_example_2() {
        int[][] prereq = {
                {1, 0},
                {0, 1}
        };
        int N = 2;
        int[] output = obj.findOrder(N, prereq);
        Assert.assertArrayEquals(new int[] {}, output);
    }

    @Test
    public void test_example_3() {
        int[][] prereq = {
                {1, 0},
                {2, 0},
                {3, 1},
                {3, 2}
        };
        int N = 4;
        int[] output = obj.findOrder(N, prereq);
        Assert.assertArrayEquals(new int[] {0,1,2,3}, output);
    }

    @Test
    public void test_my_1() {
        int N = 6;
        int[][] prereq = {
                {1, 0},
                {5, 1},
                {2, 0},
                {3, 2},
                {4, 2}
        };

        int[] output = obj.findOrder(N, prereq);
        Assert.assertArrayEquals(new int[] {0,1,2,5,3,4}, output);
    }

    @Test
    public void test_my_2() {
        int N = 6;
        int[][] prereq = {
                {1, 0},
                {5, 1},
                {2, 0},
                {3, 2},
                {4, 2},
                {5, 4}
        };

        int[] output = obj.findOrder(N, prereq);
        Assert.assertArrayEquals(new int[] {0,1,2,3,4,5}, output);
    }

    @Test
    public void test_my_3() {
        int N = 6;
        int[][] prereq = {
                {1, 0},
                {5, 1},
                {2, 0},
                {3, 2},
                {4, 2},
                {0, 4}
        };

        int[] output = obj.findOrder(N, prereq);
        Assert.assertArrayEquals(new int[] {}, output);
    }
}
