/**
 * LeetCode
 *
 * Problem 25: Reverse Nodes in k-Group
 */

package linkedlist;

public class ReverseNodesInKGroup {

    static public class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }

    static public ListNode reverseKGroup(ListNode head, int k) {
        if (k <= 1) {
            return head;
        }

        ListNode head0 = new ListNode(0);
        head0.next = head;

        ListNode p1 = head0;
        ListNode p2;

        while (true) {
            p2 = p1;
            int i = 0;
            while (i < k && p2 != null) {
                i++;
                p2 = p2.next;
            }
            if (p2 == null) {
                return head0.next;
            }

            ListNode p3 = null;
            while (p1.next != p2) {
                ListNode t = p1.next;
                p1.next = t.next;
                t.next = p2.next;
                p2.next = t;
                if (p3 == null) {
                    p3 = t;
                }
            }

            p1 = p3;
        }
    }
}
