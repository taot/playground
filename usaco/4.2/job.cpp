/*
ID:libra_k1
PROG:job
LANG:C++
*/
#include <stdio.h>
#include <string.h>
#include <queue>
#define INF 0x7f7f7f7f
#define N 1001
using namespace std;
struct job{
    int l,t;
    bool operator <(const job &x)const{return l+t>x.l+x.t;}
};
int a[N],b[N],f[N],t1[N],t2[N];
int max(int x,int y){return x>y?x:y;}
int main()
{
    freopen("job.in","r",stdin);
    freopen("job.out","w",stdout);
    int n,na,nb;
    scanf("%d%d%d",&n,&na,&nb);
    priority_queue<job>q;
    for (int i=1;i<=na;i++)
    {
        scanf("%d",&a[i]);
        q.push((job){0,a[i]});
    }
    int ans=0;
    for (int i=1;i<=n;i++)
    {
        job now=q.top();q.pop();
        ans=max(ans,now.l+now.t);
        now.l+=now.t;
        q.push(now);
        t1[i]=now.l;
    }
    while (q.size())
        q.pop();
    printf("%d ",ans);
    for (int i=1;i<=nb;i++)
    {
        scanf("%d",&b[i]);
        q.push((job){0,b[i]});
    }
    for (int i=1;i<=n;i++)
    {
        job now=q.top();q.pop();
        now.l+=now.t;
        q.push(now);
        t2[i]=now.l;
    }
    ans=0;
    for (int i=1;i<=n;i++)
        ans=max(ans,t1[i]+t2[n-i+1]);
    printf("%d\n",ans);
    return 0;
}
