/*
ID: libra_k1
LANG: JAVA
TASK: crypt1
*/
import java.io.*;
import java.util.*;

class crypt1 {

    private static String task = "crypt1";

    private static int N;

    private static int count = 0;

    private static int[] digits = null;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());

        digits = new int[N];
        st = new StringTokenizer(f.readLine());
        for (int i = 0; i < N; i++) {
            digits[i] = Integer.parseInt(st.nextToken());
        }

        int[] nums = new int[5];
        permute(0, nums);

        out.println(count);
        out.close();
    }

    private static boolean verify(int n1, int n2) {
        int[] nums2 = toDigits(n2);
        for (int i = 0; i < nums2.length; i++) {
            int p = n1 * nums2[i];
            // int[] numsp = toDigits(p);
            if (! allValidDigits(p, 3)) {
                return false;
            }
        }
        int prod = n1 * n2;
        // int[] numsprod = toDigits(prod);
        if (! allValidDigits(prod, -1)) {
            return false;
        }
        return true;
    }

    private static boolean allValidDigits(int n, int lengthLimit) {
        int[] nums = toDigits(n);
        if (lengthLimit > 0 && nums.length != lengthLimit) {
            return false;
        }
        for (int i = 0; i < nums.length; i++) {
            boolean found = false;
            for (int j = 0; j < N; j++) {
                if (digits[j] == nums[i]) {
                    found = true;
                    break;
                }
            }
            if (! found) {
                return false;
            }
        }
        return true;
    }

    private static void permute(int n, int[] nums) {
        if (n == 5) {
            int n1 = toNum(nums, 0, 3);
            int n2 = toNum(nums, 3, 2);
            // printArray(nums);
            // System.out.println(n1 + " " + n2);
            if (verify(n1, n2)) {
                // System.out.println(n1 + " " + n2);
                count++;
            }
        } else {
            for (int i = 0; i < N; i++) {
                nums[n] = digits[i];
                permute(n + 1, nums);
            }
        }
    }

    private static int toNum(int[] nums, int start, int length) {
        int n = 0;
        for (int i = 0; i < length; i++) {
            n *= 10;
            n += nums[start + i];
        }
        return n;
    }

    private static int[] toDigits(int n) {
        int[] arr = new int[10];
        int count = 0;
        while (n > 0) {
            int r = n % 10;
            n = n / 10;
            arr[count] = r;
            count++;
        }
        int[] res = new int[count];
        for (int i = 0; i < count; i++) {
            res[i] = arr[count - i - 1];
        }
        return res;
    }

    private static void printArray(int[] arr) {
        System.out.print("array: ");
        for (int i = 0; i < arr.length; i++) {
            System.out.print(" " + arr[i]);
        }
        System.out.println();
    }
}
