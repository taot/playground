/**
 * LeetCode
 *
 * 261. Graph Valid Tree
 */

package graph;

import java.util.*;

public class GraphValidTree {

    private List<Integer>[] graph;
    private boolean[] visited;

    private void makeGraph(int N, int[][] edges) {
        graph = new List[N];

        for (int i = 0; i < N; i++) {
            graph[i] = new ArrayList<Integer>();
        }

        for (int i = 0; i < edges.length; i++) {
            int from = edges[i][0];
            int to = edges[i][1];
            graph[from].add(to);
            graph[to].add(from);
        }
    }

    public boolean validTree(int N, int[][] edges) {
        if (N == 0) {
            return false;
        }

        makeGraph(N, edges);
        visited = new boolean[N];

        Queue<Integer> q = new LinkedList<>();
        q.offer(0);
        visited[0] = true;

        Integer cur = null;
        while ((cur = q.poll()) != null) {
            for (Integer next : graph[cur]) {
                if (visited[next]) {
                    return false;
                }
                q.offer(next);
                visited[next] = true;
                graph[next].remove(Integer.valueOf(cur));
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
