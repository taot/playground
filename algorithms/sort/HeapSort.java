public class HeapSort {

    public static void main(String[] args) {
        int[] data = new int[30];
        generateRandomData(data);
        System.out.println("Before sort");
        printData(data);
        long start = System.currentTimeMillis();
        heapSort(data);
        long duration = System.currentTimeMillis() - start;
        System.out.println("After sort");
        printData(data);
        System.out.println("Duration: " + duration);
    }

    static void sift(int[] data, int n, int p) {
        while (2 * p + 1 < n) {
            int c = 2 * p + 1;
            if (2 * p + 2 < n && data[2 * p + 2] > data[c]) {
                c = 2 * p + 2;
            }
            if (data[p] < data[c]) {
                swap(data, p, c);
            }
            p = c;
        }
    }

    static void heapSort(int[] data) {
        for (int i = data.length - 1; i >= 0; i--) {
            sift(data, data.length, i);
        }
        for (int n = data.length; n >= 2; n--) {
            swap(data, 0, n - 1);
            sift(data, n - 1, 0);
        }
    }

    static void swap(int[] data, int s, int t) {
        if (s == t) {
            return;
        }
        int tmp = data[s];
        data[s] = data[t];
        data[t] = tmp;
    }

    static void printData(int[] data) {
        for (int i = 0; i < data.length; i++) {
            System.out.print(data[i] + " ");
        }
        System.out.println();
    }

    static void generateRandomData(int[] data) {
        for (int i = 0; i < data.length; i++) {
            data[i] = (int) (Math.random() * 100);
        }
    }
}
