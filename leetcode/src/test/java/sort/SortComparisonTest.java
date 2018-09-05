package sort;

import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

import static sort.HeapSort.*;
import static sort.QuickSort.*;

public class SortComparisonTest {

    private MergeSort mergeSort = new MergeSort();

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
    public void test_10M() {
        int N = 10000000;
        int[] A = new int[N];
        Random rand = new Random();
        for (int i = 0; i < N; i++) {
            A[i] = rand.nextInt();
        }
        int[] B = new int[N];

        System.out.println("N = " + N);
        long start;

        System.arraycopy(A, 0, B, 0, A.length);
        start = System.currentTimeMillis();
        mergeSort.mergeSort(B);
        System.out.println("Merge sort: " + (System.currentTimeMillis() - start) + " ms");
        Assert.assertTrue(sorted(B, true));

        System.arraycopy(A, 0, B, 0, A.length);
        start = System.currentTimeMillis();
        quicksort(B);
        System.out.println("Quick sort: " + (System.currentTimeMillis() - start) + " ms");
        Assert.assertTrue(sorted(B, true));

        System.arraycopy(A, 0, B, 0, A.length);
        start = System.currentTimeMillis();
        heapsort(B);
        System.out.println("Heap sort: " + (System.currentTimeMillis() - start) + " ms");
        Assert.assertTrue(sorted(B, true));

    }
}
