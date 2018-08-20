package linkedlist;

import org.junit.Assert;
import org.junit.Test;
import static linkedlist.AddTwoNumbers.*;

public class AddTwoNumbersTest {

    @Test
    public void test_basic() {
        ListNode l1 = toListNode(342);
        printListNode(l1);
        ListNode l2 = toListNode(465);
        printListNode(l2);
        ListNode res = addTwoNumbers(l1, l2);
        printListNode(res);
        System.out.println(toNum(res));
        Assert.assertEquals(807, toNum(res));
    }
}
