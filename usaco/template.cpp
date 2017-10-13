/*
ID: libra_k1
LANG: C++
TASK: template
*/
#include <iostream>
#include <fstream>
#include <string>

std::string problem_name;

int main() {
    // read input
    problem_name = "template";
    std::ifstream fin (problem_name + ".in");
    std::ofstream fout (problem_name + ".out");

    return 0;
}
