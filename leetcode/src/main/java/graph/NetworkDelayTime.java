/**
 * LeetCode
 *
 * 743. Network Delay Time
 */

package graph;

import java.util.*;

public class NetworkDelayTime {

    class Node {
        int dst;
        int weight;
        public Node(int dst, int weight) {
            this.dst = dst;
            this.weight = weight;
        }
    }

    private List<Node>[] makeGraph(int[][] times, int N) {
        List<Node>[] graph = new List[N];
        for (int i = 0; i < N; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int i = 0; i < times.length; i++) {
            int u = times[i][0] - 1;
            int v = times[i][1] - 1;
            int w = times[i][2];
            graph[u].add(new Node(v, w));
        }
        return graph;
    }

    public int networkDelayTime(int[][] times, int N, int K) {
        List<Node>[] graph = makeGraph(times, N);
        K--;

        int[] dists = new int[N];
        for (int i = 0; i < N; i++) {
            dists[i] = -1;
        }
        dists[K] = 0;

        Set<Integer> set = new HashSet<>();

        for (int i = 0; i < N; i++) {
            int min = -1;
            int minDist = Integer.MAX_VALUE;
            for (int j = 0; j < N; j++) {
                if (set.contains(j)) {
                    continue;
                }
                if (dists[j] >= 0 && dists[j] < minDist) {
                    min = j;
                    minDist = dists[j];
                }
            }

            if (min < 0) {
                break;
            }
            set.add(min);

            for (Node node : graph[min]) {
                int j = node.dst;
                if (set.contains(j)) {
                    continue;
                }
                if (dists[j] < 0 || dists[j] > dists[min] + node.weight) {
                    dists[j] = dists[min] + node.weight;
                }
            }

        }

        int maxDist = -1;
        for (int i = 0; i < N; i++) {
            if (dists[i] < 0) {
                return -1;
            }
            if (dists[i] > maxDist) {
                maxDist = dists[i];
            }
        }

        return maxDist;
    }
}
