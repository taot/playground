/*
ID: libra_k1
LANG: JAVA
TASK: preface
*/
import java.io.*;
import java.util.*;

class preface {

    private static String task = "preface";

    static int N;

    private static TreeMap<Character, Integer> map = new TreeMap<>();
    private static List<Cell> list = new ArrayList<>();

    private final static TreeMap<Integer, String> romanMap = new TreeMap<Integer, String>();

    static {
        romanMap.put(1000, "M");
        romanMap.put(900, "CM");
        romanMap.put(500, "D");
        romanMap.put(400, "CD");
        romanMap.put(100, "C");
        romanMap.put(90, "XC");
        romanMap.put(50, "L");
        romanMap.put(40, "XL");
        romanMap.put(10, "X");
        romanMap.put(9, "IX");
        romanMap.put(5, "V");
        romanMap.put(4, "IV");
        romanMap.put(1, "I");
    }

    static String toRoman(int number) {
        int l =  romanMap.floorKey(number);
        if ( number == l ) {
            return romanMap.get(number);
        }
        return romanMap.get(l) + toRoman(number-l);
    }

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        StringTokenizer st = new StringTokenizer(f.readLine());
        N = Integer.parseInt(st.nextToken());

        for (int i = 1; i <= N; i++) {
            count(i);
        }

        for (char k : map.keySet()) {
            // out.println(k + " " + map.get(k));
            list.add(new Cell(k, map.get(k)));
        }
        Collections.sort(list, new Comparator<Cell>() {
            public int compare(Cell a, Cell b) {
                return romainMapping(a.c) - romainMapping(b.c);
            }
            private int romainMapping(char c) {
                if (c == 'I') {
                    return 1;
                }
                if (c == 'V') {
                    return 5;
                }
                if (c == 'X') {
                    return 10;
                }
                if (c == 'L') {
                    return 50;
                }
                if (c == 'C') {
                    return 100;
                }
                if (c == 'D') {
                    return 500;
                }
                if (c == 'M') {
                    return 1000;
                }
                return 0;
            }
        });
        for (Cell x : list) {
            out.println(x.c + " " + x.count);
        }

        out.close();
    }

    static void count(int n) {
        String r = toRoman(n);
        // System.out.println(r);
        for (char c : r.toCharArray()) {
            inc(c, 1);
        }
    }

    static void inc(char c, int a) {
        Integer x = map.get(c);
        if (x == null) {
            map.put(c, a);
        } else {
            map.put(c, x + a);
        }
    }

    static class Cell {
        public char c;
        public int count;
        public Cell(char c, int count) {
            this.c = c;
            this.count = count;
        }
    }
}
