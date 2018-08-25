package tree;

import org.junit.Test;

import java.util.List;

import static tree.UniqueBinarySearchTrees2.*;

public class UniqueBinarySearchTrees2Test {

    @Test
    public void test_example_1() {
        List<TreeNode> trees = generateTrees(3);
        System.out.println(trees.size());
    }
}
