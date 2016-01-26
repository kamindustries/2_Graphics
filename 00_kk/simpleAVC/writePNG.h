#include <string>
#include </usr/include/png.h>
#include <helper_math.h>
using namespace std;
///////////////////////////////////////////////////////////////////////
// code based on example code from
// http://zarb.org/~gc/html/libpng.html
///////////////////////////////////////////////////////////////////////
void writePNG(string _filename, float4* _img, int _width, int _height, int _stride) {

  // copy image data into pointers
  png_bytep* row_pointers;
  row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * _height);
  for (int y = 0; y < _height; y++) {
    row_pointers[y] = (png_byte*) malloc(sizeof(png_byte) * _width * 3);
  }
  for (int y = 0; y < _height; y++) {
    for (int x = 0; x < _width; x++) {
        int pos = (y*_height) + x;
        float4 value = _img[pos] * 255;
        value.x = (value.x > 255)  ? 255 : value.x;
        value.y = (value.y > 255)  ? 255 : value.y;
        value.z = (value.z > 255)  ? 255 : value.z;
        value.x = (value.x < 0)  ? 0 : value.x;
        value.y = (value.y < 0)  ? 0 : value.y;
        value.z = (value.z < 0)  ? 0 : value.z;
        row_pointers[_height - 1 - y][3 * x + 0] = (unsigned char)value.x;
        row_pointers[_height - 1 - y][3 * x + 1] = (unsigned char)value.y;
        row_pointers[_height - 1 - y][3 * x + 2] = (unsigned char)value.z;
    }
  }

  png_structp png_ptr;
  png_infop info_ptr;
  png_byte color_type = PNG_COLOR_TYPE_RGB;
  png_byte bit_depth = 8;

  // create file
  FILE *fp = fopen(_filename.c_str(), "wb");
  if (fp == NULL) {
    printf("[write_png_file] File %s could not be opened for writing\n", _filename.c_str());
    return;
  }

  // initialize stuff
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr)
    printf("[write_png_file] png_create_write_struct failed\n");

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    printf("[write_png_file] png_create_info_struct failed\n");

  if (setjmp(png_jmpbuf(png_ptr)))
    printf("[write_png_file] Error during init_io\n");

  png_init_io(png_ptr, fp);

  // write header
  if (setjmp(png_jmpbuf(png_ptr)))
    printf("[write_png_file] Error during writing header\n");

  png_set_compression_level(png_ptr, 7);
  png_set_IHDR(png_ptr, info_ptr, _width, _height,
       bit_depth, color_type, PNG_INTERLACE_NONE,
       PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  // write bytes
  if (setjmp(png_jmpbuf(png_ptr)))
    printf("[write_png_file] Error during writing bytes\n");

  png_write_image(png_ptr, row_pointers);


  // end write
  if (setjmp(png_jmpbuf(png_ptr)))
    printf("[write_png_file] Error during end of write\n");

  png_write_end(png_ptr, NULL);

  // cleanup heap allocation
  for (int y=0; y<_height; y++)
    free(row_pointers[y]);
  free(row_pointers);

  printf("Wrote PNG file: %s\n", _filename.c_str());

  fclose(fp);
}
