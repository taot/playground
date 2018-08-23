package linkedlist;

import org.junit.Assert;
import org.junit.Test;
import static linkedlist.RemoveNthNodeFromEndofList.*;

public class RemoveNthNodeFromEndofListTest {

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
        ListNode head = createList(new int[] { 1, 2, 3, 4, 5 });
        ListNode head2 = removeNthFromEnd(head, 2);
        Assert.assertEquals("1,2,3,5,", toString(head2));
    }

    @Test
    public void test_my_1() {
        ListNode head = createList(new int[] { 1, 2, 3, 4, 5 });
        ListNode head2 = removeNthFromEnd(head, 1);
        Assert.assertEquals("1,2,3,4,", toString(head2));
    }

    @Test
    public void test_my_2() {
        ListNode head = createList(new int[] { 1, 2, 3, 4, 5 });
        ListNode head2 = removeNthFromEnd(head, 5);
        Assert.assertEquals("2,3,4,5,", toString(head2));
    }

    @Test
    public void test_my_3() {
        ListNode head = createList(new int[] { 1 });
        ListNode head2 = removeNthFromEnd(head, 1);
        Assert.assertEquals("", toString(head2));
    }
}
