/*
ID: libra_k1
LANG: JAVA
TASK: contact
*/
import java.io.*;
import java.util.*;

class contact1 {

    private static String task = "contact";

    static int A, B, N;
    static String seq;
    static char[] chars;
    static Map<String, Integer> totalMap = new HashMap<>();
    static Map<String, Integer> currentMap = new HashMap<>();

    public static void main (String [] args) throws IOException {
        long start = System.currentTimeMillis();

        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintStream out = new PrintStream(new File(task + ".out"));
        StringTokenizer st = new StringTokenizer(f.readLine());
        A = Integer.parseInt(st.nextToken());
        B = Integer.parseInt(st.nextToken());
        N = Integer.parseInt(st.nextToken());

        StringBuilder sb = new StringBuilder();
        String s;
        while ((s = f.readLine()) != null) {
            sb.append(s);
        }
        chars = sb.toString().toCharArray();


        find();

        print(System.out);
        print(out);
        System.out.println("Duration: " + (System.currentTimeMillis() - start) + " ms");

        out.close();
        System.exit(0);
    }

    private static void print(PrintStream ps) {
        Map<Integer, List<String>> results = new TreeMap<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return -1 * o1.compareTo(o2);
            }
        });
        for (String k : totalMap.keySet()) {
            if (k.length() < A || k.length() > B) {
                continue;
            }
            int count = totalMap.get(k);
            List<String> list = results.get(count);
            if (list == null) {
                list = new ArrayList<>();
                results.put(count, list);
            }
            list.add(k);
        }
        for (Integer k : results.keySet()) {
            List<String> list = results.get(k);
            Collections.sort(list, new Comparator<String>() {
                @Override
                public int compare(String o1, String o2) {
                    if (o1.length() < o2.length()) {
                        return -1;
                    } else if (o1.length() > o2.length()) {
                        return 1;
                    } else {
                        return o1.compareTo(o2);
                    }
                }
            });
        }
        int count = 0;
        for (Integer k : results.keySet()) {
            if (count >= N) {
                break;
            }
            count++;
            ps.print(k);
            boolean first = true;
            int lineCount = 0;
            for (String s : results.get(k)) {
                if (lineCount % 6 == 0) {
                    ps.println();
                } else {
                    ps.print(" ");
                }
                ps.print(s);
                lineCount++;
            }
            ps.println();
        }
    }

    private static void find() {
        for (int i = 0; i < chars.length; i++) {
            remove(currentMap, B);
            currentMap = increment(currentMap, chars[i]);
            merge(currentMap, totalMap);
        }
    }

    private static Map<String, Integer> increment(Map<String, Integer> map, char c) {
        Map<String, Integer> newMap = new TreeMap<>();
        for (Map.Entry<String, Integer> e : map.entrySet()) {
            newMap.put(e.getKey() + c, e.getValue());
        }
        newMap.put(String.valueOf(c), 1);
        return newMap;
    }

    private static void merge(Map<String, Integer> from, Map<String, Integer> to) {
        for (Map.Entry<String, Integer> e : from.entrySet()) {
            String k = e.getKey();
            if (to.containsKey(k)) {
                to.put(k, from.get(k) + to.get(k));
            } else {
                to.put(k, from.get(k));
            }
        }
    }

    private static void remove(Map<String, Integer> map, int len) {
        Set<String> keySet = new TreeSet<>(map.keySet());
        for (String k : keySet) {
            if (k.length() >= len) {
                map.remove(k);
            }
        }
    }

    private static int encode(char c) {
        if (c == '0') {
            return 3;
        } else {
            return 1;
        }
    }

    private static int encode(int encoded, char c) {
        int n = 0;
        if (c == '1') {
            n = 1;
        }
        return (encoded << 1) + n;
    }

    private static String decode(int encoded) {
        StringBuilder sb = new StringBuilder();
        while (encoded > 1) {
            sb.append(encoded % 2);
            encoded /= 2;
        }
        return sb.reverse().toString();
    }
}
