package linkedlist;

import org.junit.Assert;
import org.junit.Test;

import static linkedlist.SwapNodesinPairs.*;

public class SwapNodesinPairsTest {

    public ListNode createList(int[] nums) {
        ListNode node = null;
        for (int i = nums.length - 1; i >= 0; i--) {
            ListNode h = new ListNode(nums[i]);
            h.next = node;
            node = h;
        }
        return node;
    }

    public String toString(ListNode node) {
        StringBuilder sb = new StringBuilder();
        while (node != null) {
            sb.append(node.val);
            sb.append(",");
            node = node.next;
        }
        return sb.toString();
    }

    @Test
    public void test_example_1() {
        ListNode l1 = createList(new int[] { 1, 2, 3, 4 });
        ListNode l2 = swapPairs(l1);
        Assert.assertEquals("2,1,4,3,", toString(l2));
    }

    @Test
    public void test_my_1() {
        ListNode l1 = createList(new int[] { 1, 2, 3, 4, 5 });
        ListNode l2 = swapPairs(l1);
        Assert.assertEquals("2,1,4,3,5,", toString(l2));
    }

    @Test
    public void test_my_2() {
        ListNode l1 = createList(new int[] {  });
        ListNode l2 = swapPairs(l1);
        Assert.assertEquals("", toString(l2));
    }
}
