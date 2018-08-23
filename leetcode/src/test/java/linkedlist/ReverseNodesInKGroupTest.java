package linkedlist;

import org.junit.Assert;
import org.junit.Test;

import static linkedlist.ReverseNodesInKGroup.*;

public class ReverseNodesInKGroupTest {

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
        ListNode l1 = createList(new int[] { 1, 2, 3, 4, 5 });
        ListNode l2 = reverseKGroup(l1, 2);
        Assert.assertEquals("2,1,4,3,5,", toString(l2));
    }

    @Test
    public void test_example_2() {
        ListNode l1 = createList(new int[] { 1, 2, 3, 4, 5 });
        ListNode l2 = reverseKGroup(l1, 3);
        Assert.assertEquals("3,2,1,4,5,", toString(l2));
    }

    @Test
    public void test_my_1() {
        ListNode l1 = createList(new int[] { 1, 2, 3, 4, 5 });
        ListNode l2 = reverseKGroup(l1, 1);
        Assert.assertEquals("1,2,3,4,5,", toString(l2));
    }

    @Test
    public void test_my_2() {
        ListNode l1 = createList(new int[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        ListNode l2 = reverseKGroup(l1, 8);
        Assert.assertEquals("8,7,6,5,4,3,2,1,", toString(l2));
    }

    @Test
    public void test_my_3() {
        ListNode l1 = createList(new int[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        ListNode l2 = reverseKGroup(l1, 4);
        Assert.assertEquals("4,3,2,1,8,7,6,5,", toString(l2));
    }

    @Test
    public void test_my_4() {
        ListNode l1 = createList(new int[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        ListNode l2 = reverseKGroup(l1, 9);
        Assert.assertEquals("1,2,3,4,5,6,7,8,", toString(l2));
    }
}
