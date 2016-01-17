#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>


static void HandleError( cudaError_t err, const char *file,  int line ) {
    if (err != cudaSuccess) {
            printf( "%s in %s at line %d\n", cudaGetErrorString( err ),  file, line );
            exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define     DIM    512

GLuint  bufferObj;
GLuint  textureID;
cudaGraphicsResource *resource;

struct sort_functor
{
  __host__ __device__
    bool operator()(uchar4 left, uchar4 right) const
    {
      return (left.y < right.y);
    }
};



// create a green/black pattern
__global__ void kernel( uchar4 *ptr ) {
// map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

// now calculate the value at that position
  float fx = x/(float)DIM - 0.5f;
  float fy = y/(float)DIM - 0.5f;
  unsigned char   green = 128 + 127 * sin( abs(fx*100) - abs(fy*100) );

// accessing uchar4 vs unsigned char*
  ptr[offset].x = 0;
  ptr[offset].y = green;
  ptr[offset].z = 0;
  ptr[offset].w = 255;
}

static void draw_func( void ) {

//  glDrawPixels( DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0 );

  glutSwapBuffers();
}
static void sort_pixels(){
  cudaGraphicsMapResources( 1, &resource, NULL );
  uchar4* devPtr;
  size_t  size;

  cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, resource);

  thrust::device_ptr<uchar4> tptr = thrust::device_pointer_cast(devPtr);
  thrust::sort(tptr, tptr+(DIM*DIM), sort_functor());
  cudaGraphicsUnmapResources( 1, &resource, NULL );
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DIM, DIM, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

  glBegin(GL_QUADS);
  glTexCoord2f( 0, 1.0f);
  glVertex3f(-1.0,1.0f,0);
  glTexCoord2f(0,0);
  glVertex3f(-1.0f,-1.0f,0);
  glTexCoord2f(1.0f,0);
  glVertex3f(1.0f,-1.0f,0);
  glTexCoord2f(1.0f,1.0f);
  glVertex3f(1.0f,1.0f,0);
  glEnd();

  draw_func();
}

static void close_func( void ){
        HANDLE_ERROR( cudaGraphicsUnregisterResource( resource ) );
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
        glDeleteBuffers( 1, &bufferObj );
        exit(0);
}

static void key_func( unsigned char key, int x, int y ) {
  switch (key) {
    case 27:
        close_func();
        break;
    case 32:
        sort_pixels();
        break;
    default:
        break;
  }
}

int main(int argc, char *argv[]) {

  cudaGLSetGLDevice( 0 );

  glutInit( &argc, argv );
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
  glutInitWindowSize( DIM, DIM );
  glutCreateWindow( "sort test" );
  glewInit();
  glGenBuffers( 1, &bufferObj );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
  glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB );

  cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone );
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, DIM, DIM, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


  cudaGraphicsMapResources( 1, &resource, NULL );
  uchar4* devPtr;
  size_t  size;

  cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, resource);
  dim3    grid(DIM/16,DIM/16);
  dim3    threads(16,16);
  kernel<<<grid,threads>>>( devPtr );
  cudaGraphicsUnmapResources( 1, &resource, NULL );

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DIM, DIM, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

  glBegin(GL_QUADS);
  glTexCoord2f( 0, 1.0f);
  glVertex3f(-1.0,1.0f,0);
  glTexCoord2f(0,0);
  glVertex3f(-1.0f,-1.0f,0);
  glTexCoord2f(1.0f,0);
  glVertex3f(1.0f,-1.0f,0);
  glTexCoord2f(1.0f,1.0f);
  glVertex3f(1.0f,1.0f,0);
  glEnd();
  draw_func();

// set up GLUT and kick off main loop
  glutCloseFunc( close_func );
  glutKeyboardFunc( key_func );
  glutDisplayFunc( draw_func );
  glutMainLoop();
}