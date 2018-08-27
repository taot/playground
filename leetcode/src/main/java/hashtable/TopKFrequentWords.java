/**
 * LeetCode
 *
 * Problem 692: Top K Frequent Words
 */

package hashtable;

import java.util.*;

public class TopKFrequentWords {

    static class Item {
        String word;
        int count;

        public Item(String word, int count) {
            this.word = word;
            this.count = count;
        }
    }

    static public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> map = new HashMap<>();
        for (String w : words) {
            Integer n = map.get(w);
            if (n == null) {
                n = 0;
            }
            n++;
            map.put(w, n);
        }

        PriorityQueue<Item> queue = new PriorityQueue<>(new Comparator<Item>() {
            @Override
            public int compare(Item o1, Item o2) {
                int d = o1.count - o2.count;
                if (d != 0) {
                    return d;
                }
                return -1 * o1.word.compareTo(o2.word);
            }
        });

        for (Map.Entry<String, Integer> ent : map.entrySet()) {
            queue.add(new Item(ent.getKey(), ent.getValue()));
            while (queue.size() > k) {
                queue.poll();
            }
        }

        List<String> output = new ArrayList<>();
        while (! queue.isEmpty()) {
            Item it = queue.poll();
            output.add(it.word);
        }
        Collections.reverse(output);

        return output;
    }
}
