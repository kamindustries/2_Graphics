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

void writeImage(float4* _data, int _increment, int _width, int _height, int _stride) {
  // float4* img_ptr = (float4*)malloc(sizeof(float4)*DIM*DIM);
  // checkCudaErrors (cudaMemcpy(img_ptr, displayPtr, sizeof(float4)*DIM*DIM, cudaMemcpyDeviceToHost ));

  // float *img_ptr = (float*)malloc(sizeof(float)*_width*_height*_stride);
  // glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, img_ptr);

  // if (_write_txt){
  //   char filename_txt[1024 * sizeof(int) / 3 + 2];
  //   sprintf(filename_txt, "data/cuda_x%d.txt", _increment);
  //   write(filename_txt, img_ptr, DIM, DIM, 4);
  // }
  // if (_write_img){
    char filename_png[2048 * sizeof(int) / 3 + 2];
    sprintf(filename_png, "data/images/velocity/cuda_x%05d.png", _increment);
    writePNG(filename_png, _data, _width, _height, _stride);
  // }
  // free(img_ptr);
}

// void writeImage(float4* _data, int _increment, int _width, int _height, int _stride) {
//   int dim = _width * _height;
//   int size = dim * _stride;
//   float* data_f = new float[size];
//
//   for (int y=0; y<_height; y++){
//     for (int x=0; x<_width; x++){
//       int pos = (y*_height) + x;
//       data_f[pos+0] = _data[pos].x;
//       data_f[pos+1] = _data[pos].y;
//       data_f[pos+2] = _data[pos].z;
//       data_f[pos+3] = _data[pos].w;
//     }
//   }
//   writeImage(data_f, _increment, _width, _height, _stride );
//   delete[] data_f;
//
//   // for (int i=0; i < dim; i++){
//     // data_f[i] = 0.0;
//     // data_f[(i*_stride)+0] = _data[i].x;
//     // data_f[(i*_stride)+1] = _data[i].y;
//     // data_f[(i*_stride)+2] = _data[i].z;
//     // data_f[(i*_stride)+3] = _data[i].w;
//   // }
//
//
// }
