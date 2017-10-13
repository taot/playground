/*
ID: libra_k1
LANG: C++
TASK: stamps
*/
#include <iostream>
#include <fstream>
#include <string>

#define MAX_N 50
#define MAX_V 2000000

std::string problem_name;

int K, N;
int stamps[MAX_N];
int nums[MAX_V+1];
int max;

int main() {
    // read input
    problem_name = "stamps";
    std::ifstream fin("stamps.in");
    std::ofstream fout("stamps.out");

    fin >> K >> N;
    max = -1;
    for (int i = 0; i < N; i++) {
        fin >> stamps[i];
        if (stamps[i] > max) {
            max = stamps[i];
        }
    }

    // DP
    nums[0] = 0;
    int i;
    for (i = 1; i <= K * max; i++) {
        int min_num = 100000;
        int found = 0;
        for (int j = 0; j < N; j++) {
            int v = i - stamps[j];
            if (v >= 0 && nums[v] < K && min_num > nums[v]) {
                min_num = nums[v];
                found = 1;
            }
        }
        if (found) {
            nums[i] = min_num + 1;
        } else {
            break;
        }
    }
    i--;

    // output
    std::cout << i << std::endl;
    fout << i << std::endl;

    return 0;
}
