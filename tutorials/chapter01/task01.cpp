#include <vector>
#include <fstream>
#include <cmath>
#include <numbers>
#include <format>

struct Vec3{float x,y,z;};
static inline Vec3 operator+(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static inline Vec3 operator-(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static inline Vec3 operator*(Vec3 a,float s){return {a.x*s,a.y*s,a.z*s};}
static inline Vec3 operator*(float s,Vec3 a){return a*s;}
static inline float dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static inline Vec3 cross(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static inline Vec3 norm(Vec3 a){float l=std::sqrt(dot(a,a));return {a.x/l,a.y/l,a.z/l};}

int main(){
    const int w=800,h=600;
    const Vec3 cam_pos={-1.2f,0.8f,1.2f};
    const Vec3 cam_look={0.5f,0.5f,0.5f};
    const Vec3 cam_up={0.0f,1.0f,0.0f};
    const float fov_y=45.0f*float(std::numbers::pi)/180.0f;
    const float aspect=float(w)/float(h);
    const Vec3 f=norm(cam_look-cam_pos);
    const Vec3 r=norm(cross(f,cam_up));
    const Vec3 u=cross(r,f);
    const float tanf=std::tan(0.5f*fov_y);

    std::ofstream fimg("out.ppm",std::ios::binary);
    if(!fimg) return 1;
    fimg<<std::format("P6\n{} {}\n255\n",w,h);
    std::vector<unsigned char> row(w*3);

    for(int y=0;y<h;++y){
        for(int x=0;x<w;++x){
            float sx=( (x+0.5f)/float(w)*2.0f-1.0f)*aspect*tanf;
            float sy=(1.0f-(y+0.5f)/float(h)*2.0f)*tanf;
            Vec3 d=norm(sx*r+sy*u+f);
            float t=0.5f*(d.y+1.0f);
            float r8=(1.0f*(1.0f-t)+0.5f*t);
            float g8=(1.0f*(1.0f-t)+0.7f*t);
            float b8=(1.0f*(1.0f-t)+1.0f*t);
            row[x*3+0]=(unsigned char)std::lround(std::fmin(1.0f,std::fmax(0.0f,r8))*255.0f);
            row[x*3+1]=(unsigned char)std::lround(std::fmin(1.0f,std::fmax(0.0f,g8))*255.0f);
            row[x*3+2]=(unsigned char)std::lround(std::fmin(1.0f,std::fmax(0.0f,b8))*255.0f);
        }
        fimg.write(reinterpret_cast<const char*>(row.data()),row.size());
    }
    return 0;
}
