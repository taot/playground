/**
 * LeetCode
 *
 * Problem 23: Merge k Sorted Lists
 */

package linkedlist;

import java.util.Comparator;
import java.util.PriorityQueue;

public class MergekSortedLists {

    static public class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }

    static public ListNode mergeKLists(ListNode[] lists) {
        ListNode head = new ListNode(0);
        ListNode tail = head;

        PriorityQueue<ListNode> queue = new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val - o2.val;
            }
        });

        for (ListNode l : lists) {
            if (l != null) {
                queue.add(l);
            }
        }

        ListNode node;

        while ((node = queue.poll()) != null) {
            tail.next = node;
            tail = node;

            if (node.next != null) {
                queue.add(node.next);
            }
        }


        return head.next;
    }
}
