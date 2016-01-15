#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <timer.h>               // timing functions
#include <time.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <helper_math.h>
#include "lodepng.h"
#include "writePNG.h"
#include "simpleGL_kernels.cuh"

#define MAX(a,b) ((a > b) ? a : b)

GLuint  bufferObj, bufferObj2;
GLuint  textureID;
// cudaGraphicsResource_t resource[2];
cudaGraphicsResource_t resource1;
cudaGraphicsResource_t resource2;
float avgFPS = 0.0f;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int frameNum = 0;
int animFrameNum = 0;
unsigned int frameCount = 0;
float *chemA, *chemB, *laplacian;
float4 *displayPtr;
bool runOnce = true;

// diffusion constants
float dA = 0.0002;
float dB = 0.00001;
float F = 0.05;
float k = 0.0675;

StopWatchInterface *timer = NULL;
timespec time1, time2;
timespec diff(timespec start, timespec end);

// mouse controls
int mouse_old_x, mouse_old_y;
bool togSimulate = false;
int max_simulate = 0;
// int pause = 17500;

bool writeDone = false;

// Convert to webm:
// png2yuv -I p -f 60 -b 1 -n 1628 -j cuda_x%05d.png > cuda_YUV.yuv
// vpxenc --good --cpu-used=0 --auto-alt-ref=1 --lag-in-frames=16 --end-usage=vbr
//        --passes=2 --threads=2 --target-bitrate=3000 -o cuda_WEBM.webm cuda_YUV.yuv
///////////////////////////////////////////////////////////////////////////////
// Functions
///////////////////////////////////////
// Write
///////////////////////////////////////
void write(const char* _filename, float4* _img) {
    FILE* file;
    file = fopen(_filename, "wb");

    int totalCells = DIM * DIM;
    // double* dataDouble = new double[totalCells * 3];
    for (int i = 0; i < totalCells; i++) {
      fprintf(file, "%f\n", _img[i].x);
      fprintf(file, "%f\n", _img[i].y);
      fprintf(file, "%f\n", _img[i].z);
    }

    fclose(file);

    // writeCpy = false;
    printf("Wrote file!\n");
}

void writeCpy (bool _write_txt = true, bool _write_img = true, int _increment = frameNum) {
  float4* img_ptr = (float4*)malloc(sizeof(float4)*DIM*DIM);
  checkCudaErrors (cudaMemcpy(img_ptr, displayPtr, sizeof(float4)*DIM*DIM, cudaMemcpyDeviceToHost ));
  if (_write_txt){
    char filename_txt[1024 * sizeof(int) / 3 + 2];
    sprintf(filename_txt, "data/cuda_x%d.txt", _increment);
    write(filename_txt, img_ptr);
  }
  if (_write_img){
    char filename_png[1024 * sizeof(int) / 3 + 2];
    sprintf(filename_png, "data/images/cuda_x%05d.png", _increment);
    writePNG(filename_png, img_ptr, DIM, DIM);
  }
}

timespec diff(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

///////////////////////////////////////
// Compute FPS
///////////////////////////////////////
void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "CUDA || %3.1f fps", avgFPS);
    glutSetWindowTitle(fps);
}


///////////////////////////////////////
// Simulate
///////////////////////////////////////
static void simulate( void ){
  if (togSimulate) {

    for (int i = 0; i < 10; i++){
      dim3    grid(DIM/16,DIM/16);
      dim3    threads(16,16);

      // *!* important
      // load chem fields with color 0,0,0,1
      if (runOnce == true){
        ClearArray<<<grid,threads>>>(chemA, 0.0);
        ClearArray<<<grid,threads>>>(chemB, 0.0);
        runOnce = false;
      }

      DrawSquare<<<grid,threads>>>(chemB);

      if (frameNum > 0) {

        Diffusion<<<grid,threads>>>( chemA, laplacian, dA, mouse_old_x, mouse_old_y );
        AddLaplacian<<<grid,threads>>>( chemA, laplacian );
        ClearArray<<<grid,threads>>>(laplacian, 0.0);

        Diffusion<<<grid,threads>>>( chemB, laplacian, dB, mouse_old_x, mouse_old_y );
        AddLaplacian<<<grid,threads>>>( chemB, laplacian );
        ClearArray<<<grid,threads>>>(laplacian, 0.0);

        React<<<grid,threads>>>( chemA, chemB );

        MakeColor<<<grid,threads>>>(chemB, displayPtr);
      }

      size_t  size;
      cudaGraphicsMapResources( 1, &resource1, 0 );
      checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&displayPtr, &size, resource1));
      // checkCudaErrors(cudaMemcpy(displayPtr, chemB, sizeof(float4)*DIM*DIM, cudaMemcpyDeviceToHost ));

      // if (frameNum == 1000 || frameNum == 9000 || frameNum == 17000 ||
      //     frameNum == 19000 || frameNum == 21000 || frameNum == 23000) {
      // if (frameNum % 500 == 0 && frameNum <= 10000) {
        // writeCpy();
      // }

      checkCudaErrors(cudaGraphicsUnmapResources( 1, &resource1, 0 ));


      frameNum++;
      // if (frameNum == pause) togSimulate = false;
    }

    // printf("chem b: %f", displayPtr[0].x);
    // printf("\r");
  }
}


///////////////////////////////////////
// Draw
///////////////////////////////////////
static void draw_func( void ) {
  // if (togSimulate) {
  //   simulate();
  // }

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);

  glClear(GL_COLOR_BUFFER_BIT);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DIM, DIM, GL_BGRA, GL_FLOAT, NULL);

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

  glutSwapBuffers();

  computeFPS();

  // cout<<diff(time1,time2).tv_sec<<":"<<diff(time1,time2).tv_nsec<<endl;
  float fr = 1.0f/60.0f;
  float df = float(diff(time1,time2).tv_nsec)/1000.0f;
  if (diff(time1,time2).tv_nsec > fr) {
    if (togSimulate) {
      writeCpy(0,1,animFrameNum);
      animFrameNum++;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    glutPostRedisplay(); // causes draw to loop forever
  }
}

///////////////////////////////////////
// Close function
///////////////////////////////////////
static void FreeResource( void ){
  // checkCudaErrors(cudaDeviceSynchronize());
  // glFinish();
  // HANDLE_ERROR( cudaGraphicsUnregisterResource(resource1));
  // HANDLE_ERROR( cudaGraphicsUnregisterResource(resourceL));
  // checkCudaErrors( cudaGraphicsUnregisterResource(resource1) );
  // cudaGraphicsUnregisterResource(resource1);
  // cudaGraphicsUnregisterResource(resource2);
  // deletePBO(&bufferObj);
  // deletePBO(&bufferObj2);
  // deletePBO(&bufferObjL);
  chemA = 0;
  chemB = 0;
  // deleteTexture(&textureID);
  // glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
  // glDeleteTextures(1, &textureID);
  glDeleteBuffers(1, &bufferObj);
  exit(0);
}

///////////////////////////////////////
// Keyboard
///////////////////////////////////////
static void key_func( unsigned char key, int x, int y ) {
  switch (key) {
    case 'q':
        FreeResource();
        break;
    case 32:
        draw_func();
        break;
    case 'p':
        togSimulate = !togSimulate;
        break;
    case '=':
        simulate();
        draw_func();
        break;
    case '.':
        writeCpy();
        break;
    default:
        break;
  }
}

void passive(int x1, int y1) {
    mouse_old_x = x1;
    mouse_old_y = y1;
    // glutPostRedisplay();

}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {

  // initialize
  glutInit( &argc, argv );
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowPosition ( 0, 0 );
  glutInitWindowSize( DIM, DIM );
  glutCreateWindow( "sort test" );
  glewInit();
  checkCudaErrors(cudaGLSetGLDevice( 0 ));


  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  // displayPtr = (float4*)malloc(sizeof(float4)*DIM*DIM);

  // on create openGL
  glGenBuffers( 1, &bufferObj );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
  glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, sizeof(float4) * DIM * DIM, NULL, GL_DYNAMIC_DRAW_ARB );
  cudaGraphicsGLRegisterBuffer( &resource1, bufferObj, cudaGraphicsMapFlagsWriteDiscard );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );

  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, DIM, DIM, 0, GL_BGRA, GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


  checkCudaErrors(cudaMalloc((void**)&chemA, sizeof(float)*DIM*DIM ));
  checkCudaErrors(cudaMalloc((void**)&chemB, sizeof(float)*DIM*DIM ));
  checkCudaErrors(cudaMalloc((void**)&laplacian, sizeof(float)*DIM*DIM ));
  checkCudaErrors(cudaMalloc((void**)&displayPtr, sizeof(float4)*DIM*DIM ));

// set up GLUT and kick off main loop
  glutCloseFunc( FreeResource );
  glutKeyboardFunc( key_func );
  glutPassiveMotionFunc(passive);
  glutIdleFunc( simulate );
  glutDisplayFunc( draw_func );
  glutMainLoop();

  cudaDeviceReset();

}
