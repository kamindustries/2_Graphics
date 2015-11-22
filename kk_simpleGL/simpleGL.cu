#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <timer.h>               // timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cutil_math.h>


// static void HandleError( cudaError_t err, const char *file,  int line ) {
//     if (err != cudaSuccess) {
//             printf( "%s in %s at line %d\n", cudaGetErrorString( err ),  file, line );
//             exit( EXIT_FAILURE );
//     }
// }
// #define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define MAX(a,b) ((a > b) ? a : b)
#define     DIM    512
#define     DT    .1

GLuint  bufferObj, bufferObj2;
GLuint  textureID;
// cudaGraphicsResource_t resource[2];
cudaGraphicsResource_t resource1;
cudaGraphicsResource_t resource2;
float ttime = 0.0f;
float avgFPS = 0.0f;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int frameNum = 0;
unsigned int frameCount = 0;
float4 *chemA, *chemB, *displayPtr;

// diffusion constants
float dA = 0.0002;
float dB = 0.00001;
float F = 0.05;
float k = 0.0675;

StopWatchInterface *timer = NULL;

// mouse controls
int mouse_old_x, mouse_old_y;


///////////////////////////////////////////////////////////////////////////////
// Functions
///////////////////////////////////////
// Sort
///////////////////////////////////////
struct sort_function
{
  __host__ __device__
    bool operator()(uchar4 left, uchar4 right) const
    {
      return (left.y < right.y);
    }
};

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
// Delete PBO
///////////////////////////////////////
void deletePBO(GLuint *pbo) 
{
    glDeleteBuffers(1, pbo);
    SDK_CHECK_ERROR_GL();
    *pbo = 0;
}

void deleteTexture(GLuint *tex)
{
    glDeleteTextures(1, tex);
    SDK_CHECK_ERROR_GL();

    *tex = 0;
}


///////////////////////////////////////
// CUDA Kernel
///////////////////////////////////////
__device__ int checkPosition(int _pos){
  int dmax = DIM*DIM;
  if (_pos < 0){
    _pos = dmax+_pos;
    return _pos % dmax;
  }
  else return _pos % dmax;
}

__global__ void Diffusion( float4 *_chem, float4 *_lap, float _difConst, bool _drawSquare, int mouse_y) {
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // q1. draws a square
  if (_drawSquare){
    float posX = (float)x/DIM;
    float posY = (float)y/DIM;

    float m_y = 1.0 - ((float)(mouse_y)/DIM);

    // if ( posX < .55 && posX > .45 && posY < .55 && posY > .45 ) {
    if ( posX < .55 && posX > .45 && posY < m_y+.05 && posY > m_y-.05 ) {    //use mouse position
      _chem[offset] = make_float4(1.,1.,1.,1.);
    }
    // else _chem[offset] = make_float4(0.,0.,0.,0.);
  }

  // constants
  float xLength = 5.12;
  float dt = 0.1;
  float dx = (float)xLength/DIM;
  float alpha = _difConst * dt / (dx*dx);

  int n1 = checkPosition((x+1) + y * blockDim.x * gridDim.x);
  int n2 = checkPosition((x-1) + y * blockDim.x * gridDim.x);
  int n3 = checkPosition(x + (y+1) * blockDim.x * gridDim.x);
  int n4 = checkPosition(x + (y-1) * blockDim.x * gridDim.x);
  // __syncthreads();

  _lap[offset] = -4.0f * _chem[offset] + _chem[n1] + _chem[n2] + _chem[n3] + _chem[n4];
  _lap[offset] *= alpha;

}

__global__ void AddLaplacian( float4 *_chem, float4 *_lap) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  _chem[offset] += _lap[offset];
  _chem[offset].w = 1.0;

}

__global__ void ReactA( float4 *_chemA, float4 *_chemB) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float F = 0.05;
  float4 A = _chemA[offset];
  float4 B = _chemB[offset];

  float4 reaction = make_float4(-A.x * (B.x*B.x) + (F * (1.0-A.x)),
                                -A.y * (B.y*B.y) + (F * (1.0-A.y)),
                                -A.z * (B.z*B.z) + (F * (1.0-A.z)),
                                1.0
                                );
  _chemA[offset] += reaction * DT;
}

__global__ void ReactB( float4 *_chemA, float4 *_chemB) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float F = 0.05;
  float k = 0.0675;
  float4 A = _chemA[offset];
  float4 B = _chemB[offset];

  float4 reaction = make_float4(A.x * (B.x*B.x) - (F+k)*B.x,
                                A.y * (B.y*B.y) - (F+k)*B.y,
                                A.z * (B.z*B.z) - (F+k)*B.z,
                                1.0
                                );
  _chemB[offset] += reaction * DT;
}

///////////////////////////////////////
// Draw
///////////////////////////////////////
static void draw_func( void ) {

  glClear(GL_COLOR_BUFFER_BIT);

  // cudaGraphicsMapResources( 1, &resource2, 0 );
  // cudaGraphicsMapResources( 1, &resourceL, 0 );
  float4 *laplacian = 0;
  size_t  size;
  // checkCudaErrors(cudaGraphicsResourceGetMappedPointer( (void**)&chemA, &size, resource1));
  // checkCudaErrors(cudaGraphicsResourceGetMappedPointer( (void**)&chemB, &size, resource2));
  // checkCudaErrors(cudaGraphicsResourceGetMappedPointer( (void**)&laplacian, &size, resourceL));
  checkCudaErrors(cudaMalloc((void**)&chemA, sizeof(float4)*DIM*DIM ));
  checkCudaErrors(cudaMalloc((void**)&chemB, sizeof(float4)*DIM*DIM ));
  checkCudaErrors(cudaMalloc((void**)&laplacian, sizeof(float4)*DIM*DIM ));

  dim3    grid(DIM/16,DIM/16);
  dim3    threads(16,16);

  Diffusion<<<grid,threads>>>( chemA, laplacian, dA, true, mouse_old_y );
  AddLaplacian<<<grid,threads>>>( chemA, laplacian );

  checkCudaErrors(cudaFree(laplacian));
  checkCudaErrors(cudaMalloc((void**)&laplacian, sizeof(float4)*DIM*DIM ));

  // Diffusion<<<grid,threads>>>( chemB, laplacian, dB, true, mouse_old_y );
  // AddLaplacian<<<grid,threads>>>( chemB, laplacian );

  // ReactA<<<grid,threads>>>( chemA, chemB );
  // ReactB<<<grid,threads>>>( chemA, chemB );

  cudaGraphicsMapResources( 1, &resource1, 0 );
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer( (void**)&displayPtr, &size, resource1));
  checkCudaErrors(cudaMemcpy(displayPtr, chemA, sizeof(float4)*DIM*DIM, cudaMemcpyDeviceToHost ));

  checkCudaErrors(cudaGraphicsUnmapResources( 1, &resource1, 0 ));
  checkCudaErrors(cudaFree(chemA));
  checkCudaErrors(cudaFree(chemB));
  checkCudaErrors(cudaFree(laplacian));

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
  ttime += 0.0001;
  frameNum++;
  glutPostRedisplay();
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
  glutInitWindowSize( DIM, DIM );
  glutCreateWindow( "sort test" );
  glewInit();
  checkCudaErrors(cudaGLSetGLDevice( 0 ));


  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  displayPtr  = 0;

  // glGenBuffers( 1, &bufferObjL );
  // glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObjL );
  // glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 32, NULL, GL_DYNAMIC_DRAW_ARB );
  // cudaGraphicsGLRegisterBuffer( &resourceL, bufferObjL, cudaGraphicsMapFlagsNone );

  // glGenBuffers( 1, &bufferObj2 );
  // glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj2 );
  // glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 32, NULL, GL_DYNAMIC_DRAW_ARB );
  // cudaGraphicsGLRegisterBuffer( &resource2, bufferObj2, cudaGraphicsMapFlagsNone );

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


// set up GLUT and kick off main loop
  glutCloseFunc( FreeResource );
  glutKeyboardFunc( key_func );
  glutPassiveMotionFunc(passive);
  glutDisplayFunc( draw_func );
  glutMainLoop();

  cudaDeviceReset();

}

