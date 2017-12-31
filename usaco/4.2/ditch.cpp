/*
ID: cmykrgb1
PROG: ditch
LANG: C++
*/
#include <iostream>
#include <fstream>
#include <string.h>
#define MAX 201
using namespace std;

class Tadjl
{
public:
    class Tnode
    {
    public:
        int r,v;
        void set(int tr,int tv)
        {
            r=tr;
            v=tv;
        }
    };
    int cnt;
    Tnode link[MAX];
};

class tQueue
{
public:
    class linklist
    {
    public:
        linklist* next;
        int value;
        linklist()
        {
            next=0;
            value=0;
        }
    };
    linklist *first,*last;
    int size;
    void add(int p)
    {
        if (size==0)
            first=last=new linklist;
        else
            last=last->next=new linklist;
        last->value=p;
        size++;
    }
    int del()
    {
        int rtn=first->value;
        linklist *tfirst=first;
        first=first->next;
        delete tfirst;
        size--;
        return rtn;
    }
    void reset()
    {
        size=0;
        first=last=0;
    }
    tQueue()
    {
        reset();
    }
};

ifstream fi("ditch.in10");
ofstream fo("ditch.out");

Tadjl adjl[MAX];
int N,M,ans;

inline int min(int a,int b)
{
    return a<b?a:b;
}

void init()
{
    int i,a,b,v;
    fi >> N >> M;
    for (i=1;i<=N;i++)
    {
        fi >> a >> b >> v;
        adjl[a].link[ ++adjl[a].cnt].set(b,v);
    }
}


int edmonds(int start,int end)
{
    int i,j,k;
    int father[MAX],fp[MAX],max[MAX];
    int Maxflow=0;
    memset(father,0,sizeof(father));
    max[start]=0x7FFFFFFF;
    tQueue *Q=new tQueue;
    Q->add(start);
    while (Q->size)
    {
        i=Q->del();
        for (k=1;k<=adjl[i].cnt;k++)
        {
            j=adjl[i].link[k].r;
            if (!adjl[i].link[k].v || j==start) continue;
            if (!father[j])
            {
                father[j]=i;
                fp[j]=k;
                max[j]=min(adjl[i].link[k].v,max[i]);
                if (j==end)
                {
                    Maxflow+=max[j];
                    while (father[j])
                    {
                        adjl[father[j]].link[fp[j]].v-=max[end];
                        adjl[j].link[++adjl[j].cnt].set(father[j],max[j]);
                        j=father[j];
                    }
                    memset(father,0,sizeof(father));
                    Q->reset();
                    Q->add(start);
                    break;
                }
                Q->add(j);
            }
        }
    }
    return Maxflow;
}

void print()
{
    fo << ans << endl;
    cout << ans << endl;
    fi.close();
    fo.close();
}

int main()
{
    init();
    ans=edmonds(1,M);
    print();
    return 0;
}
