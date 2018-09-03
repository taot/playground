/**
 * LeetCode
 *
 * Problem 207: Course Schedule
 */

package graph;

import java.util.*;

public class CourseSchedule {

    List<Integer>[] graph;
    int[] inDegrees;

    void makeGraph(int N, int[][] prerequisites) {
        graph = new List[N];
        inDegrees = new int[N];

        for (int i = 0; i < N; i++) {
            graph[i] = new ArrayList<>();
        }

        for (int i = 0; i < prerequisites.length; i++) {
            int[] p = prerequisites[i];
            int from = p[0];
            int to = p[1];
            graph[from].add(to);
            inDegrees[to] += 1;
        }
    }

    public boolean canFinish(int N, int[][] prerequisites) {
        makeGraph(N, prerequisites);
        boolean[] visited = new boolean[N];

        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < N; i++) {
            if (inDegrees[i] == 0) {
                q.offer(i);
                visited[i] = true;
            }
        }

        while (! q.isEmpty()) {
            int from = q.poll();
            for (int to : graph[from]) {
                inDegrees[to]--;
                if (inDegrees[to] == 0) {
                    if (visited[to]) {
                        return false;
                    }
                    visited[to] = true;
                    q.offer(to);
                }
            }
        }

        for (int i = 0; i < N; i++) {
            if (! visited[i]) {
                return false;
            }
        }
        return true;
    }
}
