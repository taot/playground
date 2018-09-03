/**
 * LeetCode
 *
 * Problem 210: Course Schedule II
 */

package graph;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class CourseSchedule2 {

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
            int to = p[0];
            int from = p[1];
            graph[from].add(to);
            inDegrees[to] += 1;
        }
    }

    public int[] findOrder(int N, int[][] prerequisites) {
        makeGraph(N, prerequisites);
        boolean[] visited = new boolean[N];
        List<Integer> output = new ArrayList<>();

        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < N; i++) {
            if (inDegrees[i] == 0) {
                q.offer(i);
                visited[i] = true;
                output.add(i);
            }
        }

        while (! q.isEmpty()) {
            int from = q.poll();
            for (int to : graph[from]) {
                inDegrees[to]--;
                if (inDegrees[to] == 0) {
                    if (visited[to]) {
                        return new int[0];
                    }
                    visited[to] = true;
                    q.offer(to);
                    output.add(to);
                }
            }
        }

        for (int i = 0; i < N; i++) {
            if (! visited[i]) {
                return new int[0];
            }
        }

        int[] array = new int[output.size()];
        for (int i = 0; i < output.size(); i++) {
            array[i] = output.get(i);
        }
        return array;
    }
}
