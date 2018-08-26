package sort;

public class QuickSort {

    public static void quicksort(int[] A) {
        quicksort0(A, 0, A.length);
    }

    static void quicksort0(int[] A, int start, int end) {
        if (start >= end - 1) {
            return;
        }
        int i = split(A, start, end);
        quicksort0(A, start, i);
        quicksort0(A, i+1, end);
    }

    /*
     * 闭开区间 [start, end)
     */
    static int split(int[] A, int start, int end) {
        if (start == end) {
            return start;
        }
        int x = A[end - 1];
        int i = start;
        for (int j = start; j <= end - 2; j++) {
            if (A[j] < x) {
                swap(A, i, j);
                i++;
            }
        }
        swap(A, i, end - 1);

        return i;
    }

    static void swap(int[] A, int i, int j) {
        int tmp = A[i];
        A[i] = A[j];
        A[j] = tmp;
    }
}
