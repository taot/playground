package tree;

import org.junit.Assert;
import org.junit.Test;
import static tree.UniqueBinarySearchTrees.*;

public class UniqueBinarySearchTreesTest {

    @Test
    public void test_example_1() {
        Assert.assertEquals(5, numTrees(3));
    }

    @Test
    public void test_my_1() {
        Assert.assertEquals(5, numTrees(4));
    }
}
