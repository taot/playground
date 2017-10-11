/*
ID: your_id_here
TASK: contact
LANG: C++
*/
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include <string.h>
#include <stdio.h>

#define MAX_LEN 200001

int A, B, N;
char chars[MAX_LEN];
int length;

typedef struct TrieNode {
    TrieNode *left = NULL;
    TrieNode *right = NULL;
    int count = 0;
    int sumCount = 0;
} TrieNode;

// typedef struct Holder {
//     int count = 0;
//     std::vector<std::string> *list;
// } Holder;

class my_map_key_compare {
   public:
      bool operator()(const int x,const int y) { return (x-y)>0; } // returns x>y
};

TrieNode *root;
std::map<int, std::vector<std::string>*, my_map_key_compare> result_map;

void constructTree() {
    root = new TrieNode();
    for (int i = 0; i < length; i++) {
        int len = B;
        if (length - i < B) {
            len = length - i;
        }
        TrieNode *node = root;
        for (int j = 0; j < len; j++) {
            if (chars[i + j] == '0') {
                if (node->left == NULL) {
                    node->left = new TrieNode();
                }
                node = node->left;
            } else {
                if (node->right == NULL) {
                    node->right = new TrieNode();
                }
                node = node->right;
            }
            if (j == len - 1) {
                node->count++;
            }
        }
    }
}

int sumUp(TrieNode *node) {
    if (node == NULL) {
        return 0;
    }
    int c1 = sumUp(node->left);
    int c2 = sumUp(node->right);
    node->sumCount = c1 + c2 + node->count;
    return node->sumCount;
}

void collectForPrint(TrieNode *node, std::string path) {
    if (node == NULL) {
        return;
    }
    collectForPrint(node->left, path + '0');
    collectForPrint(node->right, path + '1');
    if (path.length() < A || path.length() > B) {
        return;
    }

    std::vector<std::string> *list = result_map[node->sumCount];
    if (list == NULL) {
        list = new std::vector<std::string>();
        map[node->sumCount] = list;
    }
    list.push_back(path);
}

void print() {
    collectForPrint(root->left, "0");
    collectForPrint(root->right, "1");
}

int main() {
    // read input
    std::ofstream fout ("contact.out");
    std::ifstream fin ("contact.in");
    fin >> A >> B >> N;

    std::string s = "";
    std::string line;
    while (std::getline(fin, line)) {
        s.append(line);
    }
    length = s.length();
    strcpy(chars, s.c_str());

    // calculate
    constructTree();
    sumUp(root);
    print();

    // output

    return 0;
}
