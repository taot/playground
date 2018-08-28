package queue;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static queue.KillProcess.*;

public class KillProcessTest {

    @Test
    public void test_example_1() {
        List<Integer> pid = Arrays.asList(1, 3, 10, 5);
        List<Integer> ppid = Arrays.asList(3, 0, 5, 3);
        List<Integer> output = killProcess(pid, ppid, 5);
        List<Integer> expected = Arrays.asList(5, 10);
        Assert.assertEquals(expected, output);
    }

    @Test
    public void test_timeout_1() {
        List<Integer> pid = new ArrayList<>();
        for (int i = 1; i <= 50000; i++) {
            pid.add(i);
        }
        List<Integer> ppid = new ArrayList<>();
        ppid.add(0);
        for (int i = 1; i < 50000; i++) {
            ppid.add(1);
        }
        List<Integer> output = killProcess(pid, ppid, 1);
        for (int i = 0; i < 50000; i++) {
            Assert.assertEquals(Integer.valueOf(i+1), output.get(i));
        }
    }
}
