// OJ Problem 2569 - Rule-based handwritten digit recognition
// Implements judge(IMAGE_T&) returning 0-9 using simple CV heuristics.

#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>
#include <cmath>
using namespace std;

static vector<vector<int> > binarize(const std::vector<std::vector<double> > &img) {
    int n = (int)img.size();
    vector<vector<int> > b(n, vector<int>(n, 0));
    // Otsu-like threshold using mean; MNIST digits are white on black.
    double mean = 0.0; int cnt = 0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { mean += img[i][j]; cnt++; }
    mean /= max(1, cnt);
    double thr = max(0.2, min(0.8, mean * 0.9));
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        b[i][j] = (img[i][j] >= thr) ? 1 : 0; // 1 = white (stroke)
    }
    return b;
}

struct Box { int r0, c0, r1, c1; }; // inclusive bounds

static Box bounding_box(const vector<vector<int> > &b) {
    int n = (int)b.size();
    int r0 = n, c0 = n, r1 = -1, c1 = -1;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) if (b[i][j]) {
        r0 = min(r0, i); c0 = min(c0, j); r1 = max(r1, i); c1 = max(c1, j);
    }
    if (r1 < 0) return {0,0,n-1,n-1};
    return {r0, c0, r1, c1};
}

static vector<vector<int> > crop(const vector<vector<int> > &b, const Box &bb) {
    int h = bb.r1 - bb.r0 + 1, w = bb.c1 - bb.c0 + 1;
    vector<vector<int>> out(h, vector<int>(w, 0));
    for (int i = 0; i < h; ++i) for (int j = 0; j < w; ++j) out[i][j] = b[bb.r0+i][bb.c0+j];
    return out;
}

static int count_components(const vector<vector<int> > &b) {
    int h = (int)b.size();
    int w = (int)b[0].size();
    vector<vector<int> > vis(h, vector<int>(w, 0));
    int comps = 0; int dr[4] = {1,-1,0,0}, dc[4] = {0,0,1,-1};
    for (int i = 0; i < h; ++i) for (int j = 0; j < w; ++j) if (b[i][j] && !vis[i][j]) {
        comps++;
        queue<pair<int,int> > q; q.push(make_pair(i,j)); vis[i][j]=1;
        while(!q.empty()){
            pair<int,int> p = q.front(); q.pop();
            int r = p.first, c = p.second;
            for(int k=0;k<4;k++){int nr=r+dr[k], nc=c+dc[k];
                if(nr>=0&&nr<h&&nc>=0&&nc<w && b[nr][nc] && !vis[nr][nc]){vis[nr][nc]=1; q.push(make_pair(nr,nc));}
            }
        }
    }
    return comps;
}

static bool in_range_int(int r,int c,int h,int w){ return r>=0&&r<h&&c>=0&&c<w; }

static int count_holes(const vector<vector<int> > &b) {
    // Holes = connected components of background inside the bounding box, excluding outer background.
    int h = (int)b.size();
    int w = (int)b[0].size();
    vector<vector<int> > vis(h, vector<int>(w, 0));
    int dr[4] = {1,-1,0,0}, dc[4] = {0,0,1,-1};
    // Mark outer background via flood fill from border zeros.
    queue<pair<int,int> > q;
    for(int i=0;i<h;i++){ if(!b[i][0]){q.push(make_pair(i,0)); vis[i][0]=1;} if(!b[i][w-1]){q.push(make_pair(i,w-1)); vis[i][w-1]=1;} }
    for(int j=0;j<w;j++){ if(!b[0][j]){q.push(make_pair(0,j)); vis[0][j]=1;} if(!b[h-1][j]){q.push(make_pair(h-1,j)); vis[h-1][j]=1;} }
    while(!q.empty()){ pair<int,int> p = q.front(); q.pop(); int r=p.first, c=p.second; for(int k=0;k<4;k++){int nr=r+dr[k], nc=c+dc[k]; if(in_range_int(nr,nc,h,w) && !b[nr][nc] && !vis[nr][nc]){vis[nr][nc]=1; q.push(make_pair(nr,nc));}} }
    // Any remaining zero region is a hole.
    int holes=0;
    for(int i=0;i<h;i++) for(int j=0;j<w;j++) if(!b[i][j] && !vis[i][j]){
        holes++;
        queue<pair<int,int> > q2; q2.push(make_pair(i,j)); vis[i][j]=1;
        while(!q2.empty()){ pair<int,int> p2 = q2.front(); q2.pop(); int r=p2.first, c=p2.second; for(int k=0;k<4;k++){int nr=r+dr[k], nc=c+dc[k]; if(in_range_int(nr,nc,h,w) && !b[nr][nc] && !vis[nr][nc]){vis[nr][nc]=1; q2.push(make_pair(nr,nc));}} }
    }
    // Re-do proper flood fill for holes (fixing typo):
    fill(vis.begin(), vis.end(), vector<int>(w, 0));
    // Mark outer again
    for(int i=0;i<h;i++){ if(!b[i][0]){q.push(make_pair(i,0)); vis[i][0]=1;} if(!b[i][w-1]){q.push(make_pair(i,w-1)); vis[i][w-1]=1;} }
    for(int j=0;j<w;j++){ if(!b[0][j]){q.push(make_pair(0,j)); vis[0][j]=1;} if(!b[h-1][j]){q.push(make_pair(h-1,j)); vis[h-1][j]=1;} }
    while(!q.empty()){ pair<int,int> p3 = q.front(); q.pop(); int r=p3.first, c=p3.second; for(int k=0;k<4;k++){int nr=r+dr[k], nc=c+dc[k]; if(in_range_int(nr,nc,h,w) && !b[nr][nc] && !vis[nr][nc]){vis[nr][nc]=1; q.push(make_pair(nr,nc));}} }
    holes=0;
    for(int i=0;i<h;i++) for(int j=0;j<w;j++) if(!b[i][j] && !vis[i][j]){
        holes++;
        queue<pair<int,int> > q2; q2.push(make_pair(i,j)); vis[i][j]=1;
        while(!q2.empty()){ pair<int,int> p4 = q2.front(); q2.pop(); int r=p4.first, c=p4.second; for(int k=0;k<4;k++){int nr=r+dr[k], nc=c+dc[k]; if(in_range_int(nr,nc,h,w) && !b[nr][nc] && !vis[nr][nc]){vis[nr][nc]=1; q2.push(make_pair(nr,nc));}} }
    }
    return holes;
}

static vector<int> vproj(const vector<vector<int> > &b){
    int h=b.size(), w=b[0].size();
    vector<int> v(h,0);
    for(int i=0;i<h;i++){ int s=0; for(int j=0;j<w;j++) s+=b[i][j]; v[i]=s; }
    return v;
}
static vector<int> hproj(const vector<vector<int> > &b){
    int h=b.size(), w=b[0].size();
    vector<int> v(w,0);
    for(int j=0;j<w;j++){ int s=0; for(int i=0;i<h;i++) s+=b[i][j]; v[j]=s; }
    return v;
}

static int argmax(const vector<int>& a){ return int(max_element(a.begin(), a.end()) - a.begin()); }

int judge(std::vector<std::vector<double> > &img) {
    if (img.empty() || img[0].empty()) return 0;
    vector<vector<int> > b0 = binarize(img);
    Box bb = bounding_box(b0);
    vector<vector<int> > bc = crop(b0, bb);

    int h = (int)bc.size();
    int w = (int)bc[0].size();
    vector<int> vp = vproj(bc);
    vector<int> hp = hproj(bc);
    int holes = count_holes(bc);
    int comps = count_components(bc);

    // Fast path by holes:
    if (holes >= 2) return 8; // only 8 reliably has 2 holes
    if (holes == 1) {
        // Could be 0,6,9,4 (sometimes). Use orientation heuristics.
        int top = argmax(vp);
        int left = argmax(hp);
        // Ratio and centroid
        double mass=0, rsum=0, csum=0; for(int i=0;i<h;i++) for(int j=0;j<w;j++) if(bc[i][j]){mass++; rsum+=i; csum+=j;}
        double cr = rsum/max(1.0,mass), cc = csum/max(1.0,mass);
        double aspect = (double)h / max(1.0,(double)w);
        // 0 tends to have balanced projections and comps==1
        if (comps==1 && aspect>0.7 && aspect<1.4) return 0;
        // 9 has heavier top and right
        if (top < h/3 && left > w/2) return 9;
        // 6 has heavier bottom and left
        if (top > h/2 && left < w/2) return 6;
        // fallback
        return 0;
    }

    // No hole: {1,2,3,4,5,7}
    // Use structural heuristics.
    // 1: thin, mostly a vertical stroke with small width
    int sum=0; for(int i=0;i<h;i++) for(int j=0;j<w;j++) sum+=bc[i][j];
    double density = (double)sum / (h*w);
    int max_row = *max_element(vp.begin(), vp.end());
    int max_col = *max_element(hp.begin(), hp.end());
    double width_ratio = (double)max_col / max(1, h); // intuitive width thickness
    double height_ratio = (double)max_row / max(1, w);
    if (density < 0.12 && width_ratio < 0.35 && comps==1) return 1;

    // 7: top heavy, right-leaning, few pixels in bottom rows
    int bottom_sum = accumulate(vp.begin()+max(0,h-5), vp.end(), 0);
    int top_sum = accumulate(vp.begin(), vp.begin()+min(h,5), 0);
    if (top_sum > bottom_sum*1.8 && comps==1) return 7;

    // 4: has a crossing and two components sometimes (disconnected top-left and vertical)
    if (comps >= 2) return 4;

    // Distinguish 2/3/5 by centroid and right-side density
    int right_sum = accumulate(hp.begin()+max(0,w-5), hp.end(), 0);
    int left_sum = accumulate(hp.begin(), hp.begin()+min(w,5), 0);
    if (right_sum > left_sum*1.4 && vp.back() < vp[h/2]) return 3; // tail to the right, top heavier
    if (vp.front() > vp.back()*1.3 && left_sum > right_sum) return 2; // top-left heavy, bottom thin

    // 5: more bottom mass than top, left heavy, no hole
    if (vp.back() > vp.front()*1.2 && left_sum > right_sum) return 5;

    // Fallback by aspect/density
    if (density < 0.18) return 1;
    return 2;
}
