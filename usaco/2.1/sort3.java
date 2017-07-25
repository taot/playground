/*
ID: libra_k1
LANG: JAVA
TASK: sort3
*/
import java.io.*;
import java.util.*;

class sort3 {

    private static String task = "sort3";

    static int N;
    static int M;
    static int[] nums;
    static int[] sorted_nums;
    static int[] reduced_nums;
    static int[] reduced_sorted;
    static boolean[] grouped;
    static int G2;
    static int G3;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());
        nums = new int[N];
        sorted_nums = new int[N];

        for (int i = 0; i < N; i++) {
            st = new StringTokenizer(f.readLine());
            nums[i] = Integer.parseInt(st.nextToken());
            sorted_nums[i] = nums[i];
        }
        Arrays.sort(sorted_nums);
        M = 0;
        for (int i = 0; i < N; i++) {
            if (sorted_nums[i] != nums[i]) {
                M++;
            }
        }
        reduced_nums = new int[M];
        reduced_sorted = new int[M];
        grouped = new boolean[M];
        int idx = 0;
        for (int i = 0; i < N; i++) {
            if (sorted_nums[i] != nums[i]) {
                reduced_nums[idx] = reduced_sorted[idx] = nums[i];
                idx++;
            }
        }

        Arrays.sort(reduced_sorted);
        findGroups();

        // System.out.println(G2);
        // System.out.println(G3);
        // System.out.println(G2 + G3 * 2);
        out.println(G2 + G3 * 2);

        out.close();
    }

    static void findGroups() {
        // find group of size 2
        G2 = 0;
        while (group2()) {
        }

        // find group of size 3
        G3 = 0;
        while (group3()) {
        }
    }

    static boolean group3() {
        boolean found = false;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < M; j++) {
                for (int k = 0; k < M; k++) {
                    if (i == j || i == k || j == k) {
                        continue;
                    }
                    if (grouped[i] || grouped[j] || grouped[k]) {
                        continue;
                    }
                    if (reduced_nums[i] == reduced_sorted[j] && reduced_nums[j] == reduced_sorted[k] && reduced_nums[k] == reduced_sorted[i]) {
                        grouped[i] = grouped[j] = grouped[k] = true;
                        G3++;
                        found = true;
                    }
                }
            }
        }
        return found;
    }

    static boolean group2() {
        boolean found = false;
        for (int i = 0; i < M - 1; i++) {
            for (int j = 1; j < M; j++) {
                if (! grouped[i] && ! grouped[j] && reduced_nums[i] == reduced_sorted[j] && reduced_sorted[i] == reduced_nums[j]) {
                    grouped[i] = grouped[j] = true;
                    G2++;
                    found = true;
                }
            }
        }
        return found;
    }

    static int indexOfUngrouped() {
        for (int i = 0; i < N; i++) {
            if (! grouped[i]) {
                return i;
            }
        }
        return -1;
    }
}
