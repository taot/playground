/**
 * LeetCode
 *
 * Problem 6: ZigZag Conversion
 */

package string;

public class ZigZagConversion {

    static public String convert(String s, int nRows) {
        if (nRows <= 1) {
            return s;
        }
        int n = s.length();
        char[][] M = new char[nRows][(nRows - 1) * n / (nRows * 2 - 2) + 1];
        {
            int i = 0;
            int row = 0;
            int col = 0;
            boolean down = true;

            while (i < n) {
                if (down) {
                    if (row < nRows) {
                        M[row][col] = s.charAt(i);
                        row++;
                        i++;
                    } else {
                        down = false;
                        row -= 2;
                        col++;
                    }
                } else {
                    if (row > 0) {
                        M[row][col] = s.charAt(i);
                        row--;
                        col++;
                        i++;
                    } else {
                        down = true;
                    }
                }
            }
        }

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[0].length; j++) {
                if (M[i][j] != 0) {
                    sb.append(M[i][j]);
                }
            }
        }

        return sb.toString();
    }
}
