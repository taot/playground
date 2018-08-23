/**
 * LeetCode
 *
 * Problem 24: Swap Nodes in Pairs
 */

package linkedlist;

public class SwapNodesinPairs {

    static public class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }

    static public ListNode swapPairs(ListNode head) {
        ListNode head0 = new ListNode(0);
        head0.next = head;

        ListNode p = head0;

        while (p.next != null && p.next.next != null) {
            ListNode t = p.next;
            ListNode p2 = p.next.next;

            p.next.next = p2.next;
            p2.next = p.next;
            p.next = p2;

            p = t;
        }

        return head0.next;
    }
}
