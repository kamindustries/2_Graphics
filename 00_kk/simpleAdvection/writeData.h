#include <helper_math.h>
#include "writePNG.h"
#include "lodepng.h"

///////////////////////////////////////
// Write
///////////////////////////////////////
void write(const char* _filename, float* _img, int _width, int _height, int _stride) {
    FILE* file;
    file = fopen(_filename, "wb");
    int size = _width * _height * _stride;

    for (int i = 0; i < size; i++) {
      fprintf(file, "%f\n", _img[i]);
    }

    fclose(file);
    printf("Wrote file!\n");
}

void write(const char* _filename, float4* _img, int _width, int _height, int _stride) {
    int dim = _width * _height;
    int size = dim * _stride;
    float* data = new float[size];

    for (int i=0; i < dim; i++){
      data[i+0] = _img[i].x;
      data[i+1] = _img[i].y;
      data[i+2] = _img[i].z;
      data[i+3] = _img[i].w;
    }

    write(_filename, data, _width, _height, _stride);
    free(data);
}

void writeImage(const char *_filename, float4* _data, int _increment, int _width, int _height, int _stride) {
  char path[2048 * sizeof(int) / 3 + 2];
  snprintf(path, sizeof(path), "%s%05d.png", _filename, _increment);
  writePNG(path, _data, _width, _height, _stride);
}
