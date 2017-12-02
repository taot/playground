import java.util.Comparator;

public class fibheap {

    public static void main(String[] args) {
        int count = 10000;
//        test(count);
//        test2(count);
        test_union2(count, 300);
    }

    private static void test_union2(int count, int unionCount) {
        long start = System.currentTimeMillis();

        FibHeap<Integer> h1 = new FibHeap();
        for (int i = count - 1; i >= 0; i--) {
            h1.insert(i);
        }
        for (int j = 1; j <= unionCount; j++) {
            FibHeap h2 = new FibHeap();
            for (int i = count * (j + 1) - 1; i >= count * j; i--) {
                h2.insert(i);
            }
            h1 = h1.union(h2);

            for (int i = 0; i < count; i++) {
                Integer k = h1.extractMin();
//                System.out.println(k);
            }
        }


        long duration = System.currentTimeMillis() - start;
        System.out.println("test_union2 duration: " + duration + " ms");
    }

    private static void test_union(int count) {
        long start = System.currentTimeMillis();

        FibHeap h1 = new FibHeap();
        FibHeap h2 = new FibHeap();
        for (int i = count - 1; i >= 0; i--) {
            h1.insert(i);
        }
        for (int i = count * 2 - 1; i >= count; i--) {
            h2.insert(i);
        }

        FibHeap<Integer> h = h1.union(h2);

        for (int i = 0; i < count * 2; i++) {
            Integer k = h.extractMin();
            System.out.println(k);
        }

        long duration = System.currentTimeMillis() - start;
        System.out.println("test_union duration: " + duration + " ms");
    }

    private static void test2(int count) {
        long start = System.currentTimeMillis();

        int[] data = new int[count];
        boolean[] visisted = new boolean[count];
        for (int i = 0; i < count; i++) {
            data[i] = i;
        }

        for (int i = 0; i < count; i++) {
            int min = Integer.MAX_VALUE;
            int min_i = 0;
            for (int j = 0; j < count; j++) {
                if (! visisted[j] && data[j] < data[min_i]) {
                    min_i = j;
                    min = data[j];
                }
            }
        }

        long duration = System.currentTimeMillis() - start;
        System.out.println("test2 duration: " + duration + " ms");
    }

    private static void test(int count) {
        long start = System.currentTimeMillis();

        FibHeap<Integer> h = new FibHeap<>();
        for (int i = count - 1; i >= 0; i--) {
            h.insert(i);
        }
        for (int i = 0; i < count; i++) {
            Integer k = h.extractMin();
        }

        long duration = System.currentTimeMillis() - start;
        System.out.println("test1 duration: " + duration + " ms");
    }

    static class FibNode<E> {
        E key;
        int degree;
        FibNode p;
        FibNode child;
        FibNode left;
        FibNode right;
        boolean mark;

        @Override
        public String toString() {
            return "FibNode{" +
                    "key=" + key +
                    ", degree=" + degree +
                    ", mark=" + mark +
                    '}';
        }
    }

    static class FibHeap<E> {

        public static double LOG_PHI = Math.log((1 + Math.sqrt(5)) / 2);

        FibNode<E> min;
        int n;
        boolean destroyed;
        Comparator<E> comparator;

        public FibHeap() {
            this(null);
        }

        public FibHeap(Comparator<E> comparator) {
            this.comparator = comparator;
            this.min = null;
            this.n = 0;
        }

        public void insert(E key) {
            check();

            FibNode<E> node = new FibNode<E>();
            node.key = key;
            node.degree = 0;
            node.p = null;
            node.child = null;
            node.mark = false;

            if (this.min == null) {
                this.min = node;
                node.left = node.right = node;
            } else {
                // insert node into root list
                insertIntoList(node, this.min);

                if (compare(node.key, this.min.key) < 0) {
                    this.min = node;
                }
            }

            this.n++;
        }

        public E getMin() {
            check();
            return this.min == null ? null : this.min.key;
        }

        public FibHeap union(FibHeap<E> h) {
            check();

            FibHeap<E> nh = new FibHeap<E>();

            if (this.min == null) {
                nh.min = h.min;
            } else if (h.min == null) {
                nh.min = this.min;
            } else {
                concat(this.min, h.min);
                nh.min = compare(this.min.key, h.min.key) < 0 ? this.min : h.min;
            }

            nh.n = this.n + h.n;
            this.destroyed = h.destroyed = true;

            return nh;
        }

        public E extractMin() {
            check();

            FibNode<E> z = this.min;
            if (z == null) {
                return null;
            }

            // add z's children to root list
            if (z.child != null) {
                FibNode n = z.child;
                do {
                    n.p = null;
                    n = n.right;
                } while (n != z.child);

                concat(this.min, z.child);
            }

            // remove z from root list
            if (z == z.right) {
                this.min = null;
            } else {
                this.min = z.right;
                z.right.left = z.left;
                z.left.right = z.right;
                consolidate();
            }

            this.n -= 1;

            return z.key;
        }

        private void consolidate() {
            int D = computeD();
            FibNode<E>[] A = new FibNode[D + 1];

            FibNode w = this.min;
            if (w == null) {
                return;
            }

            FibNode sentry = w;
            do {
                // for each node w in root list
                FibNode<E> x = w;
                int d = x.degree;
                while (A[d] != null) {
                    FibNode<E> y = A[d];
                    if (compare(x.key, y.key) > 0) {
                        FibNode tmp = x;
                        x = y;
                        y = tmp;
                    }

                    // link: remove y from root list and make y a child of x
                    if (y == sentry) {
                        sentry = y.right;
                    }
                    y.right.left = y.left;
                    y.left.right = y.right;
                    if (w == y) {
                        w = y.left;
                    }
                    insertIntoList(y, x.child);
                    if (x.child == null) {
                        x.child = y;
                    }
                    y.p = x;
                    x.degree++;
                    y.mark = false;

                    A[d] = null;
                    d++;
                }
                A[d] = x;

                w = w.right;
            } while (w != sentry);

            this.min = null;
            for (int i = 0; i <= D; i++) {
                if (A[i] != null) {
                    if (this.min == null) {
                        this.min = A[i];
                        this.min.right = this.min.left = this.min;
                    } else {
                        insertIntoList(A[i], this.min);
                        if (compare(A[i].key, this.min.key) < 0) {
                            this.min = A[i];
                        }
                    }
                }
            }
        }

        private int compare(E o1, E o2) {
            if (this.comparator != null) {
                return this.comparator.compare(o1, o2);
            }
            return ((Comparable<E>) o1).compareTo(o2);
        }

        private int computeD() {
            return (int) (Math.log(this.n) / LOG_PHI);
        }

        private void concat(FibNode n1, FibNode n2) {
            // concatenate two lists
            n1.right.left = n2.left;
            n2.left.right = n1.right;
            n1.right = n2;
            n2.left = n1;
        }

        private void insertIntoList(FibNode node, FibNode list) {
            if (list == null) {
                node.right = node.left = node;
            } else {
                node.right = list.right;
                list.right.left = node;
                node.left = list;
                list.right = node;
            }
        }

        private void check() {
            if (this.destroyed) {
                throw new IllegalStateException("this fibonacci heap is already destroyed");
            }
        }
    }
}
