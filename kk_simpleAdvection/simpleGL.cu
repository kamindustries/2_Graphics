#include <stdio.h>
#include <stdlib.h>

#include "writePNG.h"
#include "simpleGL_kernels.cuh"

#define MAX(a,b) ((a > b) ? a : b)
#define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}

dim3 grid, threads;

int size = 0;
float dt = 0.1;
float diff = 0.;
float visc = 0.;
float force = 5.0;
// float source = 100.0;

GLuint  bufferObj, bufferObj2;
GLuint  textureID;
// cudaGraphicsResource_t resource[2];
cudaGraphicsResource_t resource1;
cudaGraphicsResource_t resource2;
// float ttime = 0.0f;
// float avgFPS = 0.0f;
// int fpsCount = 0;        // FPS count for averaging
// int fpsLimit = 1;        // FPS limit for sampling
int frameNum = 0;
unsigned int frameCount = 0;

float *u, *v, *u_prev, *v_prev, *source, *dens, *dens_prev;
float4 *displayPtr, *toDisplay;

bool hasRunOnce = false;

// mouse controls
int mouse_x, mouse_y, mouse_x_old, mouse_y_old;
bool togSimulate = false;
int max_simulate = 0;

bool writeCpy = false;
bool writeDone = false;

///////////////////////////////////////////////////////////////////////////////
// Initialize Variables
///////////////////////////////////////////////////////////////////////////////
void initVariables() {
  grid = dim3(DIM/16,DIM/16);
  threads = dim3(16,16);

  size = (N+2)*(N+2);
  displayPtr = (float4*)malloc(sizeof(float4)*DIM*DIM);
}

///////////////////////////////////////////////////////////////////////////////
// Initialize OpenGL
///////////////////////////////////////////////////////////////////////////////
void initGL(int argc, char *argv[]) {
  glutInit( &argc, argv );
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
  glutInitWindowSize( DIM, DIM );
  glutCreateWindow( "Simple Advection" );
  glewInit();
  glGenBuffers( 1, &bufferObj );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
  glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, sizeof(float4) * DIM * DIM, NULL, GL_DYNAMIC_DRAW_ARB );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );

  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, DIM, DIM, 0, GL_BGRA, GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void initCUDA() {
  checkCudaErrors(cudaGLSetGLDevice( 0 ));
  cudaGraphicsGLRegisterBuffer( &resource1, bufferObj, cudaGraphicsMapFlagsWriteDiscard );

  checkCudaErrors(cudaMalloc((void**)&u, sizeof(float)*size ));
  checkCudaErrors(cudaMalloc((void**)&u_prev, sizeof(float)*size ));
  checkCudaErrors(cudaMalloc((void**)&v, sizeof(float)*size ));
  checkCudaErrors(cudaMalloc((void**)&v_prev, sizeof(float)*size ));
  checkCudaErrors(cudaMalloc((void**)&dens, sizeof(float)*size ));
  checkCudaErrors(cudaMalloc((void**)&dens_prev, sizeof(float)*size ));
  checkCudaErrors(cudaMalloc((void**)&source, sizeof(float)*size ));
  checkCudaErrors(cudaMalloc((void**)&toDisplay, sizeof(float4)*size ));
}

void initArrays() {
  ClearArray<<<grid,threads>>>(u, 0.0);
  ClearArray<<<grid,threads>>>(u_prev, 0.0);
  ClearArray<<<grid,threads>>>(v, 0.0);
  ClearArray<<<grid,threads>>>(v_prev, 0.0);
  ClearArray<<<grid,threads>>>(dens, 0.0);
  ClearArray<<<grid,threads>>>(dens_prev, 0.0);
  ClearArray<<<grid,threads>>>(toDisplay, 0.0);
}

///////////////////////////////////////////////////////////////////////////////
// Simulate
///////////////////////////////////////////////////////////////////////////////
void get_from_UI(float *field, int x, int y, float value, float dt) {
  GetFromUI<<<grid,threads>>>(field, x, y, value, dt);
}

void diffuse_step(int b, float *field, float *field0, float diff, float dt){
  float a=dt*diff*N*N;
  for (int k = 0; k < 20; k++) {
    LinSolve<<<grid,threads>>>( b, field, field0, a, 1+4*a );
    SetBoundary<<<grid,threads>>>(0, field);
  }
}

void proj_step( float *u, float *v, float *p, float *div) {
    Project<<<grid,threads>>>( u, v, p, div );
    SetBoundary<<<grid,threads>>>(0, div);
    SetBoundary<<<grid,threads>>>(0, p);
    for (int k = 0; k < 20; k++) {
      LinSolve<<<grid,threads>>>( 0, p, div, 1.0, 4.0 );
      SetBoundary<<<grid,threads>>>(0, p);
    }
    ProjectFinish<<<grid,threads>>>( u, v, p, div );
    SetBoundary<<<grid,threads>>>(1, u);
    SetBoundary<<<grid,threads>>>(2, v);
}

void dens_step ( float * x, float * x0, float * u, float * v, float diff, float dt )
{
  AddSource<<<grid,threads>>>( x, x0, dt );
  SWAP ( x0, x );
  diffuse_step( 0, x, x0, diff, dt);
  SWAP ( x0, x );
  Advect<<<grid,threads>>>( 0, x, x0, u, v, dt );
  SetBoundary<<<grid,threads>>>(0, x);
}

void vel_step ( float * u, float * v, float * u0, float * v0, float visc, float dt ) {
  AddSource<<<grid,threads>>>( u, u0, dt );
  AddSource<<<grid,threads>>>( v, v0, dt );

  SWAP ( u0, u ); diffuse_step( 1, u, u0, visc, dt);
  SWAP ( v0, v ); diffuse_step( 2, v, v0, visc, dt);

  proj_step( u, v, u0, v0);

  SWAP ( u0, u );
  SWAP ( v0, v );
  Advect<<<grid,threads>>>( 1, u, u0, u0, v0, dt ); SetBoundary<<<grid,threads>>>(1, u);
  Advect<<<grid,threads>>>( 2, v, v0, u0, v0, dt ); SetBoundary<<<grid,threads>>>(2, v);

  proj_step( u, v, u0, v0);

}
static void simulate( void ){

  // *!* important
  if (!hasRunOnce) {
    initArrays();
    hasRunOnce = true;
  }

  if (frameNum > 0 && togSimulate) {
    // float x_diff = (float)(mouse_x - mouse_x_old)/float(DIM);
    // float y_diff = (float)((mouse_y_old - mouse_y))/float(DIM);
    float x_diff = mouse_x-mouse_x_old;
    float y_diff = mouse_y_old-mouse_y;
    if (frameNum % 50 == 0) printf("%f, %f\n", x_diff, y_diff);

    // ClearArray<<<grid,threads>>>(dens_prev, 0.0);
    // ClearArray<<<grid,threads>>>(u_prev, 0.0);
    // ClearArray<<<grid,threads>>>(v_prev, 0.0);

    // DrawSquare<<<grid,threads>>>(dens_prev);
    get_from_UI(dens_prev, mouse_x, DIM-mouse_y, 1.0, dt);
    get_from_UI(u_prev, mouse_x, DIM-mouse_y, x_diff*5, dt);
    get_from_UI(v_prev, mouse_x, DIM-mouse_y, y_diff*5, dt);

    // vel_step( u, v, u_prev, v_prev, visc, dt );
    dens_step( dens, dens_prev, u, v, diff, dt );

    MakeColor<<<grid,threads>>>(dens, toDisplay);
  }

  size_t  sizeT;
  cudaGraphicsMapResources( 1, &resource1, 0 );
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&displayPtr, &sizeT, resource1));
  checkCudaErrors(cudaMemcpy(displayPtr, toDisplay, sizeof(float4)*size, cudaMemcpyDeviceToHost ));

  checkCudaErrors(cudaGraphicsUnmapResources( 1, &resource1, 0 ));

  frameNum++;
  // if (frameNum == pause) togSimulate = false;

  glutPostRedisplay();
}


///////////////////////////////////////////////////////////////////////////////
// Draw
///////////////////////////////////////////////////////////////////////////////
static void draw_func( void ) {

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

  // glutPostRedisplay(); // causes draw to loop forever

}

///////////////////////////////////////////////////////////////////////////////
// Misc functions
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

  writeCpy = false;
  printf("Wrote file!\n");
}
///////////////////////////////////////
// Close
///////////////////////////////////////
static void FreeResource( void ){
  // checkCudaErrors(cudaFree(u));
  // checkCudaErrors(cudaFree(u_prev));
  // checkCudaErrors(cudaFree(v));
  // checkCudaErrors(cudaFree(v_prev));
  // checkCudaErrors(cudaFree(dens));
  // checkCudaErrors(cudaFree(dens_prev));
  // checkCudaErrors(cudaFree(source));
  // checkCudaErrors(cudaFree(toDisplay));
  glDeleteBuffers(1, &bufferObj);
}

///////////////////////////////////////
// Keyboard
///////////////////////////////////////
static void key_func( unsigned char key, int x, int y ) {
  switch (key) {
    case 'q':
    case 'Q':
        FreeResource();
        exit(0);
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
        writeCpy = true;
        break;
    case ']':
        diff += .1;
        if (diff >= 1.) diff = 1.;
        printf("Diff: %f\n", diff);
        break;
    case '[':
        diff -= .1;
        if (diff <= 0.) diff = 0.;
        printf("Diff: %f\n", diff);
        break;
    case '0':
        visc += .1;
        if (visc >= 1.) visc = 1.;
        printf("Visc: %f\n", visc);
        break;
    case '9':
        visc -= .1;
        if (visc <= 0.) visc = 0.;
        printf("Visc: %f\n", visc);
        break;
    default:
        break;
  }
}

///////////////////////////////////////////////////////////////////////////////
// GLUT Idle Func
///////////////////////////////////////////////////////////////////////////////
void passive(int x1, int y1) {
    mouse_x_old = mouse_x;
    mouse_y_old = mouse_y;
    mouse_x = x1;
    mouse_y = y1;
    // glutPostRedisplay();
}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {

  // initialize
  initVariables();
  initGL(argc, argv);
  initCUDA();

// set up GLUT and kick off main loop
  glutCloseFunc( FreeResource );
  glutKeyboardFunc( key_func );
  glutPassiveMotionFunc(passive);
  glutIdleFunc( simulate );
  glutDisplayFunc( draw_func );
  glutMainLoop();

  cudaDeviceReset();

}
