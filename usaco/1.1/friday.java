/*
ID: libra_k1
LANG: JAVA
TASK: friday
*/
import java.io.*;
import java.util.*;

class friday {

    private static String task = "friday";

    // counts array starts with Saturday
    private static int[] countsOf13thFriday = new int[] { 0, 0, 0, 0, 0, 0, 0};

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        int n = Integer.parseInt(st.nextToken());
        // n -= 1;

        int days = 0;
        for (int i = 0; i < n; i++) {
            int year = 1900 + i;
            for (int j = 1; j <= 12; j++) {
                int weekDay = getWeekDay(days + 13);
                countsOf13thFriday[weekDay] += 1;
                days += getDaysInMonth(year, j);
            }
        }
        // int weekDay = getWeekDay(days + 13);
        // countsOf13thFriday[weekDay] += 1;

        for (int i = 0; i < countsOf13thFriday.length; i++) {
            if (i != 0) {
                out.print(" ");
            }
            out.print(countsOf13thFriday[i]);
            // out.print(" ");
        }
        out.println();
        out.close();
    }

    private static int getWeekDay(int days) {
        return (days + 1) % 7;
    }

    // January is month 1
    private static int getDaysInMonth(int year, int month) {
        if (month == 4 || month == 6 || month == 9 || month == 11) {
            return 30;
        }
        if (month == 2) {
            if (isLeapYear(year)) {
                return 29;
            } else {
                return 28;
            }
        }
        return 31;
    }

    private static boolean isLeapYear(int year) {
        if (year % 400 == 0) {
            return true;
        }
        if (year % 100 == 0) {
            return false;
        }
        if (year % 4 == 0) {
            return true;
        }
        return false;
    }
}
