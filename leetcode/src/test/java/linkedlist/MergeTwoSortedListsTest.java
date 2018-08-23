package linkedlist;

import org.junit.Assert;
import org.junit.Test;

import static linkedlist.MergeTwoSortedLists.*;

public class MergeTwoSortedListsTest {

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
        ListNode l1 = createList(new int[] { 1, 2, 4 });
        ListNode l2 = createList(new int[] { 1, 3, 4 });
        ListNode l3 = mergeTwoLists(l1, l2);
        Assert.assertEquals("1,1,2,3,4,4,", toString(l3));
    }

    @Test
    public void test_my_1() {
        ListNode l1 = createList(new int[] {  });
        ListNode l2 = createList(new int[] {  });
        ListNode l3 = mergeTwoLists(l1, l2);
        Assert.assertEquals("", toString(l3));
    }

    @Test
    public void test_my_2() {
        ListNode l1 = createList(new int[] { 1,2,3,4 });
        ListNode l2 = createList(new int[] { 5,6,7 });
        ListNode l3 = mergeTwoLists(l1, l2);
        Assert.assertEquals("1,2,3,4,5,6,7,", toString(l3));
    }

    @Test
    public void test_my_3() {
        ListNode l1 = createList(new int[] { 1,4,7 });
        ListNode l2 = createList(new int[] { 2,3,5,6 });
        ListNode l3 = mergeTwoLists(l1, l2);
        Assert.assertEquals("1,2,3,4,5,6,7,", toString(l3));
    }
}
