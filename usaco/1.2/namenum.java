/*
ID: libra_k1
LANG: JAVA
TASK: namenum
*/
import java.io.*;
import java.util.*;

class namenum {

    private static String task = "namenum";

    private static PrintWriter out = null;

    private static List<String> dict = new ArrayList<String>();

    private static Map<Integer, char[]> keymap = new HashMap<Integer, char[]>();

    private static int count = 0;

    public static void main (String [] args) throws IOException {
        createKeymap();
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        int[] digits = readDigites(f);
        dict = readDict();

        char[] result = new char[digits.length];

        generate(digits, 0, result);
        if (count == 0) {
            out.println("NONE");
        }

        out.close();
    }

    private static void generate(int[] digits, int n, char[] result) {
        if (n == digits.length) {
            String s = new String(result, 0, n);
            int idx = Collections.binarySearch(dict, s);
            if (idx >= 0) {
                count++;
                out.println(s);
            }
        } else {
            int d = digits[n];
            char[] chars = keymap.get(d);
            if (chars != null) {
                for (char c : chars) {
                    result[n] = c;
                    generate(digits, n + 1, result);
                }
            }
        }
    }

    private static int[] readDigites(BufferedReader f) throws IOException {
        String s = f.readLine().trim();
        char[] chars = s.toCharArray();
        int[] digits = new int[chars.length];
        for (int i = 0; i < chars.length; i++) {
            digits[i] = chars[i] - '0';
        }
        return digits;
    }

    private static void createKeymap() {
        keymap.put(2, new char[] { 'A', 'B', 'C' });
        keymap.put(3, new char[] { 'D', 'E', 'F' });
        keymap.put(4, new char[] { 'G', 'H', 'I' });
        keymap.put(5, new char[] { 'J', 'K', 'L' });
        keymap.put(6, new char[] { 'M', 'N', 'O' });
        keymap.put(7, new char[] { 'P', 'R', 'S' });
        keymap.put(8, new char[] { 'T', 'U', 'V' });
        keymap.put(9, new char[] { 'W', 'X', 'Y' });
    }

    private static List<String> readDict() throws IOException {
        BufferedReader f = new BufferedReader(new FileReader("dict.txt"));
        List<String> list = new ArrayList<String>();

        String name = null;
        while ((name = f.readLine()) != null) {
            name = name.trim();
            if (! name.isEmpty()) {
                list.add(name);
            }
        }

        return list;
    }
}
