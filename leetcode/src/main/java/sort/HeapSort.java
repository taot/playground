package sort;

public class HeapSort {

    public static void heapsort(int[] A) {
        int n = A.length;
        heapify(A, n);
        swap(A, 0, n-1);
        for (int i = n-2; i >= 1; i--) {
            siftDown(A, 0, i+1);    // 注意这里最后一个参数是 i+1, 因为这个是 heap 的元素个数，而不是最后一个元素的下标
            swap(A, 0, i);
        }
    }

    /*
     * Loop Invariant: 保证以 i 为根的子树满足 heap 条件
     */
    public static void heapify(int[] A, int n) {
        for (int i = (n - 1) / 2; i >= 0; i--) {
            siftDown(A, i, n);
        }
    }


    public static void siftDown(int[] A, int i, int n) {
        while (2 * i + 1 <= n - 1) {
            int child = 2 * i + 1;
            if (2 * i + 2 <= n - 1 && A[2 * i + 2] > A[child]) {
                child = 2 * i + 2;
            }
            if (A[i] < A[child]) {
                swap(A, i, child);
                i = child;
            } else {
                return;
            }
        }
    }

    public static void swap(int[] A, int i, int j) {
        int tmp = A[i];
        A[i] = A[j];
        A[j] = tmp;
    }
}
