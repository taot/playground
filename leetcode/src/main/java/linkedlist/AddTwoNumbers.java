/**
 * LeetCode
 *
 * Problem 2. Add Two Numbers
 */
package linkedlist;

class AddTwoNumbers {

    static class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }

    public static ListNode toListNode(int n) {
        ListNode head = new ListNode(n % 10);
        n = n / 10;
        ListNode tail = head;
        while (n > 0) {
            ListNode node = new ListNode(n % 10);
            n = n / 10;
            tail.next = node;
            tail = node;
        }
        return head;
    }

    static public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(0);
        ListNode tail = head;
        int r = 0;
        while (l1 != null && l2 != null) {
            int v1 = l1.val;
            int v2 = l2.val;
            int v = v1 + v2 + r;
            if (v >= 10) {
                v = v - 10;
                r = 1;
            } else {
                r = 0;
            }
            tail.next = new ListNode(v);
            tail = tail.next;
            l1 = l1.next;
            l2 = l2.next;
        }
        ListNode remain = l1;
        if (l1 == null) {
            remain = l2;
        }
        while (remain != null) {
            int v1 = remain.val;
            int v = v1 + r;
            if (v >= 10) {
                v = v - 10;
                r = 1;
            } else {
                r = 0;
            }
            tail.next = new ListNode(v);
            tail = tail.next;
            remain = remain.next;
        }
        if (r > 0) {
            tail.next = new ListNode(r);
            tail = tail.next;
        }
        return head.next;
    }

    static int toNum(ListNode l) {
        int n = 0;
        int base = 1;
        while (l != null) {
            n += base * l.val;
            base *= 10;
            l = l.next;
        }
        return n;
    }

    static void printListNode(ListNode l) {
        while (l != null) {
            System.out.print(l.val + " ");
            l = l.next;
        }
        System.out.println();
    }
}