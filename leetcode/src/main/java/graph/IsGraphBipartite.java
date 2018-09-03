/**
 * LeetCode
 *
 * 785. Is Graph Bipartite ?
 */

package graph;

public class IsGraphBipartite {

    private int[] color;

    private boolean dfs(int[][] graph, int k, int clr) {
        if (color[k] != 0) {
            return color[k] == clr;
        }

        color[k] = clr;

        for (int m : graph[k]) {
            if (! dfs(graph, m, -clr)) {
                return false;
            }
        }
        return true;
    }

    public boolean isBipartite(int[][] graph) {
        int N = graph.length;
        color = new int[N];

        for (int i = 0; i < N; i++) {
            if (color[i] == 0 && ! dfs(graph, i, 1)) {
                return false;
            }
        }

        return true;
    }
}
