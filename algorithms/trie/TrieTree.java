import java.util.*;

public class TrieTree {

    public static void main(String[] args) {
        Trie trie = new Trie();

        insert(trie, "Terry");
        insert(trie, "Tree");
        insert(trie, "Terrific");
        insert(trie, "Terrible");
        insert(trie, "Trend");

        List<String> ss = traverse(trie, "");
        printList("after insert", ss);
        ss = search(trie, "Terry");
        printList("search", ss);
        remove(trie, "Terry");
        remove(trie, "Tree");
        remove(trie, "Terrific");
        remove(trie, "Terrible");
        remove(trie, "Trend");
        ss = traverse(trie, "");
        printList("remove", ss);
        System.out.println("## count: " + countNodes(trie));
    }

    static void insert(Trie trie, String s) {
        char[] chars = s.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            char c = chars[i];
            Trie t = trie.map.get(c);
            if (t == null) {
                t = new Trie();
                trie.map.put(c, t);
            }
            if (i == chars.length - 1) {
                t.isEndpoint = true;
            }
            trie = t;
        }
    }

    // static void remove(Trie trie, String s) {
    //     char[] chars = s.toCharArray();
    //     for (int i = 0; i < chars.length; i++) {
    //         char c = chars[i];
    //         Trie t = trie.map.get(c);
    //         if (t == null) {
    //             break;
    //         }
    //         if (i == chars.length - 1) {
    //             t.isEndpoint = false;
    //         }
    //         trie = t;
    //     }
    // }

    static void remove(Trie trie, String s) {
        char[] chars = s.toCharArray();
        removeRecursive(trie, chars, 0);
    }

    static boolean removeRecursive(Trie trie, char[] chars, int end) {
        if (end == chars.length) {
            trie.isEndpoint = false;
        } else {
            char c = chars[end];
            Trie t = trie.map.get(c);
            if (t != null) {
                boolean toRemove = removeRecursive(t, chars, end + 1);
                if (toRemove) {
                    trie.map.remove(c);
                }
            }
        }
        return ! trie.isEndpoint && trie.map.isEmpty();
    }

    // s is the prefix
    static List<String> search(Trie trie, String s) {
        char[] chars = s.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            char c = chars[i];
            Trie t = trie.map.get(c);
            if (t == null) {
                return Collections.emptyList();
            }
            trie = t;
        }
        return traverse(trie, s);
    }

    static List<String> traverse(Trie trie, String prefix) {
        List<String> results = new ArrayList<>();
        char[] chars = new char[100];
        traverseRecursive(trie, prefix, chars, 0, results);
        return results;
    }

    static void traverseRecursive(Trie trie, String prefix, char[] chars, int end, List<String> results) {
        // if (trie == null) {
        //     return;
        // }
        if (trie.isEndpoint) {
            char[] chars2 = new char[end];
            System.arraycopy(chars, 0, chars2, 0, end);
            String s = new String(chars);
            results.add(prefix + s);
        }
        for (char k : trie.map.keySet()) {
            Trie t = trie.map.get(k);
            chars[end] = k;
            traverseRecursive(t, prefix, chars, end + 1, results);
        }
    }

    static class Trie {
        public boolean isEndpoint = false;
        public Map<Character, Trie> map = new HashMap<Character, Trie>();
    }

    static void printList(String head, List<String> lst) {
        System.out.println("## " + head + ":");
        for (String s : lst) {
            System.out.println(s);
        }
    }

    static int countNodes(Trie trie) {
        int sum = 1;
        for (char k : trie.map.keySet()) {
            Trie t = trie.map.get(k);
            sum += countNodes(t);
        }
        return sum;
    }
}
