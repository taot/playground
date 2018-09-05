package sort;

public class MergeSort {

    private int[] B;

    void recursive(int[] A, int start, int end) {
        if (start >= end - 1) {
            return;
        }
        int mid = (start + end) / 2;
        recursive(A, start, mid);
        recursive(A, mid, end);
        merge(A, start, mid, end);
    }

    void merge(int[] A, int start, int mid, int end) {
        int i = start;
        int j = mid;
        int k = start;
        while (i < mid && j < end) {
            if (A[i] <= A[j]) {
                B[k] = A[i];
                i++;
            } else {
                B[k] = A[j];
                j++;
            }
            k++;
        }
        while (i < mid) {
            B[k++] = A[i++];
        }
        while (j < end) {
            B[k++] = A[j++];
        }
        System.arraycopy(B, start, A, start, end - start);
    }

    void mergeSort(int[] A) {
        B = new int[A.length];
        recursive(A, 0, A.length);
    }
}
