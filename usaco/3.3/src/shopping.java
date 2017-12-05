/*
ID: libra_k1
LANG: JAVA
TASK: shopping
*/
import java.io.*;
import java.util.*;

class shopping {

    private static String task = "shopping";

    static int nProds = 0;
    static int[] nProdItems = new int[5];
    static int[] prices = new int[5];
    static int nOffers = 0;
    static int[][] offers = new int[104][5];
    static int[] offerPrices = new int[104];

    static int[][][][][] lowPrices = new int[6][6][6][6][6];

    public static void main (String [] args) throws IOException {
        readInputs();
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));

        dp();

        int i0, i1, i2, i3, i4;
        int minPrice = lowPrices [nProdItems[0]] [nProdItems[1]] [nProdItems[2]] [nProdItems[3]] [nProdItems[4]];
        System.out.println(minPrice);
        out.println(minPrice);

        out.close();
    }

    static void dp() {
        for (int i0 = 0; i0 <= 5; i0++) {
            for (int i1 = 0; i1 <= 5; i1++) {
                for (int i2 = 0; i2 <= 5; i2++) {
                    for (int i3 = 0; i3 <= 5; i3++) {
                        for (int i4 = 0; i4 <= 5; i4++) {
                            int min = Integer.MAX_VALUE;
                            for (int k = 0; k < nOffers + nProds; k++) {
                                int[] offer = offers[k];
                                if (i0 - offer[0] < 0 || i1 - offer[1] < 0 || i2 - offer[2] < 0 || i3 - offer[3] < 0 || i4 - offer[4] < 0) {
                                    continue;
                                }
                                int sum = lowPrices [i0 - offer[0]] [i1 - offer[1]] [i2 - offer[2]] [i3 - offer[3]] [i4 - offer[4]];
                                sum += offerPrices[k];
                                if (sum < min) {
                                    min = sum;
                                }
                            }
                            if (min != Integer.MAX_VALUE) {
                                lowPrices[i0][i1][i2][i3][i4] = min;
                            }
                        }
                    }
                }
            }
        }
    }

    static void readInputs() throws IOException {
        // init
        for (int i = 0; i < 5; i++) {
            prices[i] = 100;
        }

        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        StringTokenizer st = new StringTokenizer(f.readLine());
        int s = Integer.parseInt(st.nextToken());
        // skip
        for (int i = 0; i < s; i++) {
            f.readLine();
        }
        st = new StringTokenizer(f.readLine());
        nProds = Integer.parseInt(st.nextToken());
        int nextIdx = 0;
        Map<Integer, Integer> codeIdx = new HashMap<Integer, Integer>();
        for (int i = 0; i < nProds; i++) {
            st = new StringTokenizer(f.readLine());
            int c = Integer.parseInt(st.nextToken());
            int k = Integer.parseInt(st.nextToken());
            int p = Integer.parseInt(st.nextToken());
            Integer idx = codeIdx.get(c);
            if (idx == null) {
                idx = nextIdx;
                codeIdx.put(c, idx);
                nextIdx++;
            }
            nProdItems[idx] = k;
            prices[idx] = p;
        }
        f.close();
        if (codeIdx.size() > 5) {
            throw new RuntimeException("num of products greater than 5");
        }

        // re-read file
        f = new BufferedReader(new FileReader(task + ".in"));
        st = new StringTokenizer(f.readLine());
        s = Integer.parseInt(st.nextToken());
        outer:
        for (int i = 0; i < s; i++) {
            st = new StringTokenizer(f.readLine());
            int n = Integer.parseInt(st.nextToken());
            for (int j = 0; j < n; j++) {
                int c = Integer.parseInt(st.nextToken());
                int k = Integer.parseInt(st.nextToken());
                Integer idx = codeIdx.get(c);
                if (idx == null) {
                    continue outer;
                }
                offers[nOffers][idx] = k;
            }
            int p = Integer.parseInt(st.nextToken());
            offerPrices[nOffers] = p;
            nOffers++;
        }
        for (int i = 0; i < nProds; i++) {
            offers[nOffers + i][i] = 1;
            offerPrices[nOffers + i] = prices[i];
        }
        f.close();
    }
}
