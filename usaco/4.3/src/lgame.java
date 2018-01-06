/*
ID: libra_k1
LANG: JAVA
TASK: lgame
*/
import java.io.*;
import java.util.*;

class lgame {

    private static String task = "lgame";

    static Map<Character, Integer> valueMap = new HashMap<>();

    static {
        valueMap.put('a', 2);
        valueMap.put('b', 5);
        valueMap.put('c', 4);
        valueMap.put('d', 4);
        valueMap.put('e', 1);
        valueMap.put('f', 6);
        valueMap.put('g', 5);
        valueMap.put('h', 5);
        valueMap.put('i', 1);
        valueMap.put('j', 7);
        valueMap.put('k', 6);
        valueMap.put('l', 3);
        valueMap.put('m', 5);
        valueMap.put('n', 2);
        valueMap.put('o', 3);
        valueMap.put('p', 5);
        valueMap.put('q', 7);
        valueMap.put('r', 2);
        valueMap.put('s', 1);
        valueMap.put('t', 2);
        valueMap.put('u', 4);
        valueMap.put('v', 6);
        valueMap.put('w', 6);
        valueMap.put('x', 7);
        valueMap.put('y', 5);
        valueMap.put('z', 7);
    }

    static List<Word> dict = new ArrayList<>();
    static List<Word> dict2 = new ArrayList<>();
    static List<List<Word>> results = new ArrayList<>();
    static int maxValue;

    public static void main (String [] args) throws IOException {
        BufferedReader f = new BufferedReader(new FileReader(task + ".in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(task + ".out")));
        readDict();
        String line = f.readLine();
        Word input = new Word(line);
        filterDict(input);
        maxValue = 0;

//        System.out.println(dict.size());
//        System.out.println(dict2.size());

        recursive(input.chars, 0, null);

//        System.out.println(results.size());
        System.out.println(maxValue);
        out.println(maxValue);

        Set<String[]> set = removeDup();
        for (String[] words : set) {
            for (int i = 0; i < words.length; i++) {
                if (i > 0) {
                    out.print(' ');
                    System.out.print(' ');
                }
                out.print(words[i]);
                System.out.print(words[i]);
            }
            System.out.println();
            out.println();
        }

        out.close();
    }

    static Set<String[]> removeDup() {
        Set<String[]> set = new TreeSet<>(new Comparator<String[]>() {
            @Override
            public int compare(String[] words1, String[] words2) {
                int ml = Math.max(words1.length, words2.length);
                for (int i = 0; i < ml; i++) {
                    int c = words1[i].compareTo(words2[i]);
                    if (c != 0) {
                        return c;
                    }
                }
                return words1.length - words2.length;
            }
        });
        for (List<Word> l : results) {
            Set<String> ts = new TreeSet<>();
            for (Word w : l) {
                ts.add(w.s);
            }
            set.add(ts.toArray(new String[ts.size()]));
        }
        return set;
    }

    static void recursive(int[] chars, int value, Node words) {
        if (value > maxValue) {
            results.clear();
            maxValue = value;
            results.add(words.toList());
        } else if (value == maxValue && value != 0) {
            results.add(words.toList());
        }
        for (Word w : dict2) {
            if (w.isWordValid(chars)) {
                int[] chars2 = new int[26];
                for (int i = 0; i < 26; i++) {
                    chars2[i] = chars[i] - w.chars[i];
                }
                recursive(chars2, value + w.value, new Node(w, words));
                for (int i = 0; i < 26; i++) {
                    chars2[i] = chars[i] - w.chars[i];
                }
            }
        }
    }

    static void filterDict(Word input) {
        for (Word w : dict) {
            if (w.isWordValid(input.chars)) {
                dict2.add(w);
            }
        }
    }

    static void readDict() throws IOException {
        BufferedReader f = new BufferedReader(new FileReader("lgame.dict"));
        String l = null;
        while (! (l = f.readLine()).equals(".")) {
            dict.add(new Word(l));
        }
        f.close();
    }

    static class Word {
        public String s;
        public int[] chars = new int[26];
        public int value;

        public Word(String s) {
            this.s = s;
            char[] arr = s.toCharArray();
            int v = 0;
            for (char c : arr) {
                chars[c - 'a']++;
                v += valueMap.get(c);
            }
            this.value = v;
        }

        public boolean isWordValid(int[] chars) {
            for (char c = 'a'; c <= 'z'; c++) {
                if (! isValid(c, chars[c - 'a'])) {
                    return false;
                }
            }
            return true;
        }

        public boolean isValid(char c, int count) {
            return chars[c - 'a'] <= count;
        }
    }

    static class Node {
        public Word word;
        public Node prev;

        public Node(Word word, Node prev) {
            this.word = word;
            this.prev = prev;
        }

        public List<Word> toList() {
            Node n = this;
            List<Word> list = new ArrayList<>();
            while (n != null) {
                list.add(n.word);
                n = n.prev;
            }
            return list;
        }
    }
}
