import java.util.PriorityQueue;

public class Test {

    public static void main(String[] args) {
        PriorityQueue<M> q = new PriorityQueue<>();
        q.add(new M());
        q.add(new M());
        q.poll();
    }

    static class M {
        int key;
    }
}
