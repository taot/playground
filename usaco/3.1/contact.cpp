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
#include <algorithm>

#include <string.h>
#include <stdio.h>

#define MAX_LEN 200001

int A, B, N;
char chars[MAX_LEN];
int length;

typedef struct TrieNode {
    TrieNode *left;
    TrieNode *right;
    int count;
    int sumCount;
} TrieNode;

class my_map_key_compare {
   public:
      bool operator()(const int x,const int y) { return (x-y)>0; } // returns x>y
};

TrieNode *root;
std::map<int, std::vector<std::string>*, my_map_key_compare> result_map;

void initTrieNode(TrieNode *n) {
    n->left = NULL;
    n->right = NULL;
    n->count = 0;
    n->sumCount = 0;
}

void constructTree() {
    root = new TrieNode();
    initTrieNode(root);
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
                    initTrieNode(node->left);
                }
                node = node->left;
            } else {
                if (node->right == NULL) {
                    node->right = new TrieNode();
                    initTrieNode(node->right);
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
    collectForPrint(node->left, path + "0");
    collectForPrint(node->right, path + "1");
    if (path.length() < A || path.length() > B) {
        return;
    }

    std::vector<std::string> *list = result_map[node->sumCount];
    if (list == NULL) {
        list = new std::vector<std::string>();
        result_map[node->sumCount] = list;
    }
    list->push_back(path);
}

bool my_string_cmp(const std::string &s1, const std::string &s2) {
    int l1 = s1.length();
    int l2 = s2.length();
    if (l1 < l2) {
        return 1;
    } else if (l1 > l2) {
        return 0;
    }
    return (s1 < s2);
}

void print(std::ostream &out) {
    collectForPrint(root->left, "0");
    collectForPrint(root->right, "1");
    int count = 0;
    for (std::map<int, std::vector<std::string>*>::iterator it = result_map.begin(); it != result_map.end(); it++) {
        count++;
        if (count > N) {
            break;
        }
        int k = it->first;
        out << k;
        std::vector<std::string> *list = it->second;
        std::sort(list->begin(), list->end(), my_string_cmp);
        int count2 = 0;
        for (std::vector<std::string>::iterator it2 = list->begin(); it2 != list->end(); it2++) {
            if (count2 % 6 == 0) {
                out << std::endl;
            } else {
                out << " ";
            }
            out << *it2;
            count2++;
        }
        out << std::endl;
    }
}

void my_append(char *chars, int &length, std::string &line) {
    for (std::string::iterator it = line.begin(); it != line.end(); it++) {
        char c = *it;
        if (c != ' ' && c != '\n' && c != '\r') {
            chars[length] = c;
            length++;
        }
    }
}

int main() {
    // read input
    std::ifstream fin ("contact.in");
    std::ofstream fout ("contact.out");
    fin >> A >> B >> N;

    std::string s = "";
    std::string line;
    length = 0;
    while (std::getline(fin, line)) {
        my_append(chars, length, line);
    }

    // calculate
    constructTree();
    sumUp(root);

    // output
    // print(std::cout);
    print(fout);

    return 0;
}
