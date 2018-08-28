/**
 * LeetCode
 *
 * Problem 582: Kill Process
 */

package queue;

import java.util.*;

public class KillProcess {

    static public List<Integer> killProcess(List<Integer> pid, List<Integer> ppid, int kill) {
        // pre-process: make map ppid -> pid
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < pid.size(); i++) {
            int ch = pid.get(i);
            int p = ppid.get(i);
            List<Integer> list = map.get(p);
            if (list == null) {
                list = new ArrayList<>();
                map.put(p, list);
            }
            list.add(ch);
        }

        List<Integer> list = new ArrayList<>();
        Queue<Integer> q = new ArrayDeque<>();

        q.offer(kill);

        while (! q.isEmpty()) {
            int p = q.poll();
            list.add(p);
            List<Integer> children = map.get(p);
            if (children != null) {
                for (int c : children) {
                    q.offer(c);
                }
            }
        }

        Collections.sort(list);
        return list;
    }
}
