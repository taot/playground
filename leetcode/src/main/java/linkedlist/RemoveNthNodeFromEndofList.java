/**
 * LeetCode
 *
 * Problem 19: Remove Nth Node From End of List
 */

package linkedlist;

public class RemoveNthNodeFromEndofList {

    static class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }

    static public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode head0 = new ListNode(0);
        head0.next = head;
        head = head0;

        ListNode p1 = head;
        ListNode p2 = null;

        do {
            if (n == 0) {
                p2 = head;
            }

            p1 = p1.next;
            if (p2 != null) {
                p2 = p2.next;
            }
            n--;
        } while (p1.next != null);

        if (n == 0) {
            p2 = head;
        }

        p2.next = p2.next.next;

        return head.next;
    }
}
