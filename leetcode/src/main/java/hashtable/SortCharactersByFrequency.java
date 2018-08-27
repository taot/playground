/**
 * LeetCode
 *
 * Problem 451: Sort Characters By Frequency
 */

package hashtable;

import java.util.*;

public class SortCharactersByFrequency {

    static class Item {
        char c;
        int count;

        public Item(char c, int count) {
            this.c = c;
            this.count = count;
        }
    }

    static public String frequencySort(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()) {
            Integer n = map.get(c);
            if (n == null) {
                n = 0;
            }
            n++;
            map.put(c, n);
        }

        List<Item> list = new ArrayList<>();
        for (char c : map.keySet()) {
            int count = map.get(c);
            list.add(new Item(c, count));
        }

        list.sort(new Comparator<Item>() {
            @Override
            public int compare(Item o1, Item o2) {
                int d = o2.count - o1.count;
                if (d != 0) {
                    return d;
                }
                return o1.c - o2.c;
            }
        });

        StringBuilder sb = new StringBuilder();
        for (Item it : list) {
            for (int i = 0; i < it.count; i++) {
                sb.append(it.c);
            }
        }

        return sb.toString();
    }
}
