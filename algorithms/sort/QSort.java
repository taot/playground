public class QSort {

    public static void main(String[] args) {
        int[] data = new int[5000000];
        generateRandomData(data);
        System.out.println("Before sort");
//        printData(data);
        long start = System.currentTimeMillis();
        qsort(data, 0, data.length-1);
        long duration = System.currentTimeMillis() - start;
        System.out.println("After sort");
//        printData(data);
        System.out.println("Duration: " + duration);
    }
    
    static int partition(int[] data, int s, int t) {
        int x = data[t];
        int i = s - 1;
        for (int j = s; j < t; j++) {
            if (data[j] < x) {
                i++;
                swap(data, i, j);
            }
        }
        swap(data, i+1, t);
        return i+1;
    }

    static void qsort(int[] data, int s, int t) {
        if (s >= t) {
            return;
        }
        int p = partition(data, s, t);
        qsort(data, s, p-1);
        qsort(data, p+1, t);
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
            data[i] = (int) (Math.random() * 1000000);
        }
    }
}
