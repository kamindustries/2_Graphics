#include <stdio.h>
#include <stdlib.h>

#include "writePNG.h"
#include "simpleGL_kernels.cuh"

#define MAX(a,b) ((a > b) ? a : b)
#define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}

dim3 grid, threads;

int size = 0;
int win_x = 512;
int win_y = 512;
float dt = 0.1;
float diff = 0.;
float visc = 0.;
float force = 5.0;
float source_density = 100.0;

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
float *dens_cpu;
float4 *displayPtr, *toDisplay;

bool hasRunOnce = false;

// mouse controls
static int mouse_down[3];
int mouse_x, mouse_y, mouse_x_old, mouse_y_old;
bool togSimulate = true;
int max_simulate = 0;

bool writeCpy = false;
bool writeDone = false;


int ID(int i, int j) { return (i+((N+2)*j)); }

///////////////////////////////////////////////////////////////////////////////
// Initialize Variables
///////////////////////////////////////////////////////////////////////////////
void initVariables() {
  grid = dim3(DIM/16,DIM/16);
  threads = dim3(16,16);

  size = (N+2)*(N+2);
  displayPtr = (float4*)malloc(sizeof(float4)*DIM*DIM);
  dens_cpu = (float*)malloc(sizeof(float)*size);
}

///////////////////////////////////////////////////////////////////////////////
// Initialize OpenGL
///////////////////////////////////////////////////////////////////////////////
void initGL(int argc, char *argv[]) {
  glutInit( &argc, argv );
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
  glutInitWindowPosition ( 0, 0 );
  // glutInitWindowSize( DIM, DIM );
  glutInitWindowSize ( win_x, win_y );
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

  glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
	glClear ( GL_COLOR_BUFFER_BIT );
	glutSwapBuffers ();
	glClear ( GL_COLOR_BUFFER_BIT );
	glutSwapBuffers ();

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
// Sim steps
///////////////////////////////////////////////////////////////////////////////
void get_from_UI(float *d, float *u, float *v) {

  int i, j = (N+2)*(N+2);

  WeirdThing<<<grid,threads>>>(d, u, v); // rename this

  if ( !mouse_down[0] && !mouse_down[2] ) return;

  // map mouse position to window size
  i = (int)((mouse_x /(float)win_x)*N+1);
	j = (int)(((win_y-mouse_y)/(float)win_y)*N+1);

  float x_diff = mouse_x-mouse_x_old;
  float y_diff = mouse_y_old-mouse_y;
  if (frameNum % 50 == 0) printf("%f, %f\n", x_diff, y_diff);

  if ( i<1 || i>N || j<1 || j>N ) return;

  // DrawSquare<<<grid,threads>>>(dens_prev);

  if ( mouse_down[0] ) {
    GetFromUI<<<grid,threads>>>(u, i, j, x_diff * force);
    GetFromUI<<<grid,threads>>>(v, i, j, y_diff * force);
  }

  if ( mouse_down[2]) {
    GetFromUI<<<grid,threads>>>(d, i, j, source_density);
  }

  mouse_x_old = mouse_x;
  mouse_y_old = mouse_y;

  return;
}

void diffuse_step(int b, float *field, float *field0, float diff, float dt){
  float a=dt*diff*float(N)*float(N); // needed to float(N) to get it to work...
  for (int k = 0; k < 20; k++) {
    LinSolve<<<grid,threads>>>( b, field, field0, a, (float)1.0+(4.0*a) );
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

///////////////////////////////////////////////////////////////////////////////
// Simulate
///////////////////////////////////////////////////////////////////////////////
static void simulate( void ){

  // *!* important
  if (!hasRunOnce) {
    initArrays();
    hasRunOnce = true;
  }

  if (frameNum > 0 && togSimulate) {
    get_from_UI(dens_prev, u_prev, v_prev);
    vel_step( u, v, u_prev, v_prev, visc, dt );
    dens_step( dens, dens_prev, u, v, diff, dt );
    MakeColor<<<grid,threads>>>(dens, toDisplay);
  }

  size_t  sizeT;
  cudaGraphicsMapResources( 1, &resource1, 0 );
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&displayPtr, &sizeT, resource1));
  checkCudaErrors(cudaMemcpy(displayPtr, toDisplay, sizeof(float4)*size, cudaMemcpyDeviceToHost ));
  checkCudaErrors(cudaGraphicsUnmapResources( 1, &resource1, 0 ));

  checkCudaErrors(cudaMemcpy(dens_cpu, dens, sizeof(float)*size, cudaMemcpyDeviceToHost ));

  frameNum++;
  glutPostRedisplay();
}



static void pre_display ( void )
{
	glViewport ( 0, 0, win_x, win_y );
	glMatrixMode ( GL_PROJECTION );
	glLoadIdentity ();
	gluOrtho2D ( 0.0, 1.0, 0.0, 1.0 );
	glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
	glClear ( GL_COLOR_BUFFER_BIT );
}

///////////////////////////////////////////////////////////////////////////////
// Draw
///////////////////////////////////////////////////////////////////////////////
static void draw_func( void ) {

  glViewport ( 0, 0, win_x, win_y );
  glMatrixMode ( GL_PROJECTION );
  glLoadIdentity ();
  gluOrtho2D ( 0.0, 1.0, 0.0, 1.0 );

  glClear(GL_COLOR_BUFFER_BIT);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DIM, DIM, GL_BGRA, GL_FLOAT, NULL);

  glBegin(GL_QUADS);
  glTexCoord2f( 0, 1.0f);
  glVertex3f(0.0,1.0,0.0);
  glTexCoord2f(0,0);
  glVertex3f(0.0,0.0,0.0);
  glTexCoord2f(1.0f,0);
  glVertex3f(1.0f,0.0,0.0);
  glTexCoord2f(1.0f,1.0f);
  glVertex3f(1.0,1.0,0.0);
  glEnd();

  // pre_display ();

  // int i, j;
  // float x, y, h, d00, d01, d10, d11;
  //
  // h = 1.0f/float(N);
  //
  // glBegin ( GL_QUADS );
  //   for ( i=0 ; i<=N ; i++ ) {
  //     x = (float)(i-0.5f)*h;
  //     for ( j=0 ; j<=N ; j++ ) {
  //       y = (float)(j-0.5f)*h;
  //
  //       d00 = dens_cpu[ID(i,j)];
  //       d01 = dens_cpu[ID(i,j+1)];
  //       d10 = dens_cpu[ID(i+1,j)];
  //       d11 = dens_cpu[ID(i+1,j+1)];
  //
  //       glColor4f ( d00, d00, d00, 1.0 ); glVertex2f ( x, y );
  //       glColor4f ( d10, d10, d10, 1.0 ); glVertex2f ( x+h, y );
  //       glColor4f ( d11, d11, d11, 1.0 ); glVertex2f ( x+h, y+h );
  //       glColor4f ( d01, d01, d01, 1.0 ); glVertex2f ( x, y+h );
  //     }
  //   }
  // glEnd ();

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
// GLUT Mouse
///////////////////////////////////////////////////////////////////////////////
void motion_func(int x, int y) {
  mouse_x = x;
  mouse_y = y;
}

void mouse_func ( int button, int state, int x, int y )
{
	mouse_x_old = mouse_x = x;
	mouse_y_old = mouse_x = y;

	mouse_down[button] = state == GLUT_DOWN;
}

static void reshape_func ( int width, int height )
{
	// glutSetWindow ( win_id );
	glutReshapeWindow ( width, height );

	win_x = width;
	win_y = height;
}
///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {

  // initialize
  initVariables();
  initGL(argc, argv);
  initCUDA();

  // pre_display ();


// set up GLUT and kick off main loop
  // glutCloseFunc( FreeResource );
  glutKeyboardFunc( key_func );
  glutMouseFunc ( mouse_func );
  glutMotionFunc(motion_func);
  glutIdleFunc( simulate );
  glutReshapeFunc ( reshape_func );
  glutDisplayFunc( draw_func );
  glutMainLoop();

  cudaDeviceReset();

}
