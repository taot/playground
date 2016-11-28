/*
ID: libra_k1
LANG: JAVA
TASK: ride
*/
import java.io.*;
import java.util.*;

class ride {
    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader("ride.in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("ride.out")));

        String line = f.readLine();
        int prod1 = prod(line);

        line = f.readLine();
        int prod2 = prod(line);


        if (prod1 == prod2) {
            out.println("GO");
        } else {
            out.println("STAY");
        }


        out.close();
    }

    private static int prod(String s) {
        char[] chars = s.toCharArray();
        int prod = 1;
        for (char c : chars) {
            prod *= (c - 'A' + 1);
        }
        return prod % 47;
    }
}
