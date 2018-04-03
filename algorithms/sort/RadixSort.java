public class RadixSort {
    
    public static void main(String[] args) {
        int[] data = new int[5000000];
        generateRandomData(data);
        System.out.println("Before sort");
//        printData(data);
        long start = System.currentTimeMillis();
        radixSort(data);
        long duration = System.currentTimeMillis() - start;
        System.out.println("After sort");
//        printData(data);
        System.out.println("Duration: " + duration);
    }

    static void countRadix(int[] counts, int[] data, int radix) {
        for (int i = 0; i < counts.length; i++) {
            counts[i] = 0;
        }
        for (int i = 0; i < data.length; i++) {
            int r = getRadix(data[i], radix);
            counts[r]++;
        }
        for (int i = 1; i < counts.length; i++) {
            counts[i] += counts[i-1];
        }
        for (int i = counts.length - 1; i >= 1; i--) {
            counts[i] = counts[i-1];
        }
        counts[0] = 0;
    }

    static void radixSort(int[] data) {
        int[] tmpData = new int[data.length];
        int[] counts = new int[10];
        for (int i = 0; i < 8; i++) {
            countRadix(counts, data, i);
            for (int j = 0; j < data.length; j++) {
                int r = getRadix(data[j], i);
                tmpData[counts[r]] = data[j];
                counts[r]++;
            }
            for (int j = 0; j < data.length; j++) {
                data[j] = tmpData[j];
            }
        }
    }

    static int getRadix(int n, int radix) {
        for (int j = 0; j < radix; j++) {
            n /= 10;
        }
        int r = n % 10;
        return r;
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
