package graph;

import org.junit.Assert;
import org.junit.Test;

public class GraphValidTreeTest {

    GraphValidTree algo = new GraphValidTree();

    @Test
    public void test_example_1() {
        int N = 5;
        int[][] input = {{0,1}, {0,2}, {0,3}, {1,4}};
        boolean output = algo.validTree(N, input);
        Assert.assertEquals(true, output);
    }

    @Test
    public void test_example_2() {
        int N = 5;
        int[][] input = {{0,1}, {1,2}, {2,3}, {1,3}, {1,4}};
        boolean output = algo.validTree(N, input);
        Assert.assertEquals(false, output);
    }

    @Test
    public void test_my_1() {
        int N = 1;
        int[][] input = {};
        boolean output = algo.validTree(N, input);
        Assert.assertEquals(true, output);
    }

    @Test
    public void test_my_2() {
        int N = 0;
        int[][] input = {};
        boolean output = algo.validTree(N, input);
        Assert.assertEquals(false, output);
    }

    @Test
    public void test_my_3() {
        int N = 2;
        int[][] input = {{0,1}, {1,0}};
        boolean output = algo.validTree(N, input);
        Assert.assertEquals(false, output);
    }

    @Test
    public void test_my_4() {
        int N = 3;
        int[][] input = {};
        boolean output = algo.validTree(N, input);
        Assert.assertEquals(false, output);
    }

    @Test
    public void test_wrong_1() {
        int N = 3;
        int[][] input = {{1,0}, {2,0}};
        boolean output = algo.validTree(N, input);
        Assert.assertEquals(true, output);
    }
}
