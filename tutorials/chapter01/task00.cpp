#include <vector>
#include <fstream>

int main() {
    const int w = 800, h = 600;
    std::ofstream f("out.ppm", std::ios::binary);
    if (!f) return 1;
    f << "P6\n" << w << " " << h << "\n255\n";
    std::vector<unsigned char> row(w * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float u = (x + 0.5f) / static_cast<float>(w);
            float v = (y + 0.5f) / static_cast<float>(h);
            row[x * 3 + 0] = static_cast<unsigned char>(u * 255.0f);
            row[x * 3 + 1] = static_cast<unsigned char>(v * 255.0f);
            row[x * 3 + 2] = static_cast<unsigned char>(51);
        }
        f.write(reinterpret_cast<const char*>(row.data()), row.size());
    }
    return 0;
}
