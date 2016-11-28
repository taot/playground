/*
ID: libra_k1
LANG: JAVA
TASK: template
*/
import java.io.*;
import java.util.*;

class template {

    private static String task = "template";

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        int i1 = Integer.parseInt(st.nextToken());
        int i2 = Integer.parseInt(st.nextToken());
        out.println(i1+i2);
        out.close();
    }
}
