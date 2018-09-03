package graph;

import org.junit.Assert;
import org.junit.Test;

public class IsGraphBipartiteTest {

    IsGraphBipartite algo = new IsGraphBipartite();

    @Test
    public void test_example_1() {
        int[][] input = {{1,3}, {0,2}, {1,3}, {0,2}};
        boolean output = algo.isBipartite(input);
        Assert.assertEquals(true, output);
    }

    @Test
    public void test_example_2() {
        int[][] input = {{1,2,3}, {0,2}, {0,1,3}, {0,2}};
        boolean output = algo.isBipartite(input);
        Assert.assertEquals(false, output);
    }

    @Test
    public void test_my_1() {
        int[][] input = {{1}, {0}};
        boolean output = algo.isBipartite(input);
        Assert.assertEquals(true, output);
    }

    @Test
    public void test_my_2() {
        int[][] input = {{1}, {0,2}, {1}};
        boolean output = algo.isBipartite(input);
        Assert.assertEquals(true, output);
    }

    @Test
    public void test_wrong_1() {
        int[][] input = {{2,3,5,6,7,8,9},{2,3,4,5,6,7,8,9},{0,1,3,4,5,6,7,8,9},{0,1,2,4,5,6,7,8,9},{1,2,3,6,9},{0,1,2,3,7,8,9},{0,1,2,3,4,7,8,9},{0,1,2,3,5,6,8,9},{0,1,2,3,5,6,7},{0,1,2,3,4,5,6,7}};
        boolean output = algo.isBipartite(input);
        Assert.assertEquals(false, output);
    }
}
