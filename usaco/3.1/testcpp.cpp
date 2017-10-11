#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>

void test_string() {
    std::string s = "1234";
    std::cout << s << std::endl;
    std::string s1 = s + '3';
    std::cout << s << std::endl;
    std::cout << s1 << std::endl;
}

template <typename T>
void print_vector(std::vector<T> *v) {
    for(typename std::vector<T>::iterator it = v->begin(); it != v->end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
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

bool my_map_key_cmp(const int &a, const int &b) {
    return a > b;
}

class my_map_key_compare {
   public:
      bool operator()(const int x,const int y) { return (x-y)>0; } // returns x>y
};

void test_map() {
    // std::map<int, std::string> mymap;
    // mymap.insert(std::pair<int, std::string>(1, "hello"));
    // mymap.insert(std::pair<int, std::string>(2, "hi"));
    // std::string s = mymap[0];
    // if (s == "") {
    //     std::cout << "is null" << std::endl;
    // }

    // std::map<int, int> map2;
    // map2[1] = 1;
    // map2[1] = 2;
    // map2[2] = 2;
    // int a = map2[-1];
    // std::cout << map2.count(0) << std::endl;

    std::map<int, std::vector<std::string>*, my_map_key_compare> map3;

    map3[1] = new std::vector<std::string>();
    map3[1]->push_back("1110");
    map3[1]->push_back("00");
    map3[1]->push_back("11");
    map3[1]->push_back("10");
    map3[1]->push_back("110");
    map3[1]->push_back("01100");
    map3[1]->push_back("100");

    map3[2] = new std::vector<std::string>();
    map3[2]->push_back("01110");
    map3[2]->push_back("000");
    map3[2]->push_back("011");

    // std::vector<std::string> *v = map3[1];
    // std::sort(v->begin(), v->end(), my_string_cmp);
    // print_vector(map3[1]);
    for (std::map<int, std::vector<std::string>*>::iterator it = map3.begin(); it != map3.end(); it++) {
        std::cout << it->first << std::endl;
        std::sort(it->second->begin(), it->second->end(), my_string_cmp);
        print_vector(it->second);
    }

    if (map3[3] == NULL) {
        std::cout << "is null" << std::endl;
    } else {
        std::cout << "not null" << std::endl;
    }
    // std::cout << "get: " << map3. << std::endl;
}

int main() {
    test_map();
    return 0;
}
