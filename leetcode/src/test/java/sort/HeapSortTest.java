package sort;

import org.junit.Assert;
import org.junit.Test;

import static sort.HeapSort.heapsort;

public class HeapSortTest {

    public void printArray(int[] A) {
        for (int a : A) {
            System.out.print(a + " ");
        }
        System.out.println();
    }

    public boolean sorted(int[] A, boolean ascend) {
        boolean flag = true;
        for (int i = 0; i < A.length - 1; i++) {
            if (ascend) {
                flag = flag && A[i] <= A[i+1];
            } else {
                flag = flag && A[i] >= A[i+1];
            }
        }
        return flag;
    }

    @Test
    public void test_my_1() {
        int[] A = { 1, 3, 4, 2 };
        heapsort(A);
        printArray(A);
        Assert.assertTrue(sorted(A, true));
    }

    @Test
    public void test_my_2() {
        int[] A = { 1, 3, 4, 2, 0, -1, 10, 5, 3 };
        heapsort(A);
        printArray(A);
        Assert.assertTrue(sorted(A, true));
    }
}
