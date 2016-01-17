#include <stdio.h>
#include <stdlib.h>

#include "writePNG.h"
#include "simpleGL_kernels.cuh"

#define MAX(a,b) ((a > b) ? a : b)
#define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}
#define REFRESH_DELAY     10 //ms

dim3 grid, threads;

int size = 0;
int win_x = 512;
int win_y = 512;
float dt = .1;
float diff = 0.0f;
float visc = 0.0f;
float force = 5.0;
float source_density = 10.0;
// diffusion constants
float dA = 0.0002;
float dB = 0.00001;

GLuint  bufferObj, bufferObj2;
GLuint  textureID;
cudaGraphicsResource_t resource1;

float avgFPS = 0.0f;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int frameNum = 0;
int animFrameNum = 0;

StopWatchInterface *timer = NULL;
timespec time1, time2;
timespec time_diff(timespec start, timespec end);

float *u, *v, *u_prev, *v_prev, *dens, *dens_prev;
float *chemA, *chemA_prev, *chemB, *chemB_prev,  *laplacian;
float4 *displayPtr, *toDisplay;

bool hasRunOnce = false;

// mouse controls
static int mouse_down[3];
int mouse_x, mouse_y, mouse_x_old, mouse_y_old;
bool togSimulate = false;
int max_simulate = 0;

int drawChem = 0;


int ID(int i, int j) { return (i+((N+2)*j)); }

// Convert to webm:
// png2yuv -I p -f 60 -b 1 -n 1628 -j cuda_x%05d.png > cuda_YUV.yuv
// vpxenc --good --cpu-used=0 --auto-alt-ref=1 --lag-in-frames=16 --end-usage=vbr --passes=2 --threads=2 --target-bitrate=3000 -o cuda_WEBM.webm cuda_YUV.yuv

///////////////////////////////////////////////////////////////////////////////
// Initialize Variables
///////////////////////////////////////////////////////////////////////////////
void initVariables() {
  grid = dim3(DIM/16,DIM/16);
  threads = dim3(16,16);

  size = (N+2)*(N+2);
  displayPtr = (float4*)malloc(sizeof(float4)*DIM*DIM);

  // Create the CUTIL timer
  sdkCreateTimer(&timer);
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
  checkCudaErrors(cudaMalloc((void**)&chemA, sizeof(float)*size ));
  checkCudaErrors(cudaMalloc((void**)&chemA_prev, sizeof(float)*size ));
  checkCudaErrors(cudaMalloc((void**)&chemB, sizeof(float)*size ));
  checkCudaErrors(cudaMalloc((void**)&chemB_prev, sizeof(float)*size ));
  checkCudaErrors(cudaMalloc((void**)&toDisplay, sizeof(float4)*size ));
  checkCudaErrors(cudaMalloc((void**)&laplacian, sizeof(float)*size ));
}

void initArrays() {
  ClearArray<<<grid,threads>>>(u, 0.0);
  ClearArray<<<grid,threads>>>(u_prev, 0.0);
  ClearArray<<<grid,threads>>>(v, 0.0);
  ClearArray<<<grid,threads>>>(v_prev, 0.0);
  ClearArray<<<grid,threads>>>(dens, 0.0);
  ClearArray<<<grid,threads>>>(dens_prev, 0.0);
  ClearArray<<<grid,threads>>>(chemA, 1.0);
  ClearArray<<<grid,threads>>>(chemA_prev, 1.0);
  ClearArray<<<grid,threads>>>(chemB, 0.0);
  ClearArray<<<grid,threads>>>(chemB_prev, 0.0);
  ClearArray<<<grid,threads>>>(toDisplay, 0.0);
  ClearArray<<<grid,threads>>>(laplacian, 0.0);
}

void computeFPS() {
  frameNum++;
  fpsCount++;

  if (fpsCount == fpsLimit) {
    avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
    fpsCount = 0;
    fpsLimit = (int)MAX(avgFPS, 1.f);

    sdkResetTimer(&timer);
  }

  char fps[256];
  sprintf(fps, "Cuda GL Interop: %3.1f fps (Max 100Hz)", avgFPS);
  glutSetWindowTitle(fps);
}

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
    char filename_png[2048 * sizeof(int) / 3 + 2];
    sprintf(filename_png, "data/images/cuda_x%05d.png", _increment);
    writePNG(filename_png, img_ptr, DIM, DIM);
  }
  free(img_ptr);
}

timespec time_diff(timespec start, timespec end)
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


///////////////////////////////////////////////////////////////////////////////
// Sim steps
///////////////////////////////////////////////////////////////////////////////
void get_from_UI(float *_chemA, float *_chemB, float *u, float *v) {

  int i, j = (N+2)*(N+2);

  // which is faster?
  // ClearThreeArrays<<<grid,threads>>>(d, u, v);
  ClearArray<<<grid,threads>>>(_chemA, 1.0);
  ClearArray<<<grid,threads>>>(_chemB, 0.0);
  ClearArray<<<grid,threads>>>(u, 0.0);
  ClearArray<<<grid,threads>>>(v, 0.0);

  DrawSquare<<<grid,threads>>>(_chemB, 1.0);
  // DrawSquare<<<grid,threads>>>(chemB, 1.0);

  if ( !mouse_down[0] && !mouse_down[2] ) return;

  // map mouse position to window size
  i = (int)((mouse_x /(float)win_x)*N+1);
	j = (int)(((win_y-mouse_y)/(float)win_y)*N+1);

  float x_diff = mouse_x-mouse_x_old;
  float y_diff = mouse_y_old-mouse_y;
  if (frameNum % 50 == 0) printf("%f, %f\n", x_diff, y_diff);

  if ( i<1 || i>N || j<1 || j>N ) return;

  if ( mouse_down[0] ) {
    GetFromUI<<<grid,threads>>>(u, i, j, x_diff * force);
    GetFromUI<<<grid,threads>>>(v, i, j, y_diff * force);
  }

  if ( mouse_down[2]) {
    GetFromUI<<<grid,threads>>>(_chemB, i, j, source_density);
  }

  mouse_x_old = mouse_x;
  mouse_y_old = mouse_y;

  return;
}

void diffuse_step(int b, float *field, float *field0, float diff, float dt){
  float a=dt*diff*float(N)*float(N); // needed to float(N) to get it to work...
  for (int k = 0; k < 20; k++) {
    LinSolve<<<grid,threads>>>( field, field0, a, (float)1.0+(4.0*a) );
    SetBoundary<<<grid,threads>>>( b, field );
  }
}

void RD_step(int b, float *chemA, float *chemA0, float *chemB, float *chemB0, float diff, float dt){
  float a1=dt*dA*float(N)*float(N); // needed to float(N) to get it to work...
  float a2=dt*dB*float(N)*float(N); // needed to float(N) to get it to work...

  for (int k = 0; k < 20; k++) {
    LinSolve<<<grid,threads>>>( chemA, chemA0, a1, (float)1.0+(4.0*a1) );
    LinSolve<<<grid,threads>>>( chemB, chemB0, a2, (float)1.0+(4.0*a2) );
    SetBoundary<<<grid,threads>>>( b, chemA );
    SetBoundary<<<grid,threads>>>( b, chemB );
  }
}

void advect_step ( int b, float *field, float *field0, float *u, float *v, float dt ){
  Advect<<<grid,threads>>>( field, field0, u, v, dt );
  SetBoundary<<<grid,threads>>>( b, field );
}

void proj_step( float *u, float *v, float *p, float *div) {
    Project<<<grid,threads>>>( u, v, p, div );
    SetBoundary<<<grid,threads>>>(0, div);
    SetBoundary<<<grid,threads>>>(0, p);
    for (int k = 0; k < 20; k++) {
      LinSolve<<<grid,threads>>>( p, div, 1.0, 4.0 );
      SetBoundary<<<grid,threads>>>(0, p);
    }
    ProjectFinish<<<grid,threads>>>( u, v, p, div );
    SetBoundary<<<grid,threads>>>(1, u);
    SetBoundary<<<grid,threads>>>(2, v);
}

// void dens_step ( float *chemA, float *chemA0, float *u, float *v, float diff, float dt )
// {
//   AddSource<<<grid,threads>>>(chemA,chemA0, dt );
//   SWAP (chemA0,chemA );
//   diffuse_step( 0,chemA,chemA0, diff, dt);
//
//   SWAP (chemA0,chemA );
//   advect_step(0,chemA,chemA0, u, v, dt);
// }

void dens_step ( float *chemA, float *chemA0, float *chemB, float *chemB0,
                  float *u, float *v, float diff, float dt )
{

  // Naive ARD-----------------------
  AddSource<<<grid,threads>>>(chemB, chemB0, dt );
  chemA0 = chemA;
  chemB0 = chemB;
  for (int i = 0; i < 10; i++){
    Diffusion<<<grid,threads>>>(chemA, laplacian, dA, dt);
    AddLaplacian<<<grid,threads>>>(chemA, laplacian);
    ClearArray<<<grid,threads>>>(laplacian, 0.0);

    Diffusion<<<grid,threads>>>(chemB, laplacian, dB, dt);
    AddLaplacian<<<grid,threads>>>(chemB, laplacian);
    ClearArray<<<grid,threads>>>(laplacian, 0.0);

    React<<<grid,threads>>>( chemA, chemB, dt );
  }

  // // Diffusion Ted-----------------------
  // AddSource<<<grid,threads>>>(chemB, chemB0, dt );
  // chemA0 = chemA;
  // chemB0 = chemB;
  // ReactStable<<<grid,threads>>>( chemA, chemB, dt );
  // SWAP (chemA0,chemA );
  // SWAP (chemB0,chemB );
  // // diffuse_step( 0,chemA,chemA0, dA, dt);
  // // diffuse_step( 0,chemB,chemB0, dB, dt);
  //
  // float a1=dt*dA*float(N)*float(N);
  // float a2=dt*dB*float(N)*float(N);
  // float h = (float)1.0/float(N);
  // float h1 = 1.0+(4.0*a1);
  // float h2 = 1.0+(4.0*a2);
  // for (int i = 0; i < 20; i++){
  //   DiffusionTed<<<grid,threads>>>(chemA, chemA0, dt, a1, h1);
  //   DiffusionTed<<<grid,threads>>>(chemB, chemB0, dt, a2, h2);
  //   SetBoundary<<<grid,threads>>>( 0, chemA );
  //   SetBoundary<<<grid,threads>>>( 0, chemB );
  // }

  // float *chemA_tmp = chemA;
  // float *chemB_tmp = chemB;

  // ReactStable<<<grid,threads>>>( chemA_tmp, chemB_tmp, dt );

  // SWAP (chemA0,chemA );
  // SWAP (chemB0,chemB );

  // RD_step( 0,chemA,chemA0, chemB, chemB0, diff, dt);

  // Advect-----------------------
  SWAP (chemA0,chemA );
  advect_step(0,chemA,chemA0, u, v, dt);
  SWAP (chemB0,chemB );
  advect_step(0,chemB,chemB0, u, v, dt);


  // React<<<grid,threads>>>( chemA, chemB, dt );
}


void vel_step ( float *u, float *v, float *u0, float *v0, float visc, float dt ) {
  AddSource<<<grid,threads>>>( u, u0, dt );
  AddSource<<<grid,threads>>>( v, v0, dt );

  SWAP ( u0, u ); diffuse_step( 1, u, u0, visc, dt);
  SWAP ( v0, v ); diffuse_step( 2, v, v0, visc, dt);

  proj_step( u, v, u0, v0);

  SWAP ( u0, u );
  SWAP ( v0, v );
  advect_step(1, u, u0, u0, v0, dt);
  advect_step(2, v, v0, u0, v0, dt);

  proj_step( u, v, u0, v0);
}

///////////////////////////////////////////////////////////////////////////////
// Simulate
///////////////////////////////////////////////////////////////////////////////
static void simulate( void ){
  sdkStartTimer(&timer);

  // *!* important
  if (!hasRunOnce) {
    initArrays();
    hasRunOnce = true;
  }

  if (frameNum > 0 && togSimulate) {
    get_from_UI(chemA_prev, chemB_prev, u_prev, v_prev);
    vel_step( u, v, u_prev, v_prev, visc, dt );
    dens_step( chemA, chemA_prev, chemB, chemB_prev, u, v, diff, dt );
    if (drawChem == 0) MakeColor<<<grid,threads>>>(chemB, displayPtr);
    else if (drawChem == 1) MakeColor<<<grid,threads>>>(chemA, displayPtr);
  }

  size_t  sizeT;
  cudaGraphicsMapResources( 1, &resource1, 0 );
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&displayPtr, &sizeT, resource1));
  // checkCudaErrors(cudaMemcpy(displayPtr, toDisplay, sizeof(float4)*size, cudaMemcpyDeviceToHost ));
  checkCudaErrors(cudaGraphicsUnmapResources( 1, &resource1, 0 ));

  sdkStopTimer(&timer);
  computeFPS();
  // glutPostRedisplay();
}


static void pre_display ( void ) {
  glViewport ( 0, 0, win_x, win_y );
  glMatrixMode ( GL_PROJECTION );
  glLoadIdentity ();
  gluOrtho2D ( 0.0, 1.0, 0.0, 1.0 );
  glClear(GL_COLOR_BUFFER_BIT);
}

///////////////////////////////////////////////////////////////////////////////
// Draw
///////////////////////////////////////////////////////////////////////////////
static void draw_func( void ) {

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
  pre_display ();

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

  glutSwapBuffers();

  float fr = 1.0f/60.0f;
  float df = float(time_diff(time1,time2).tv_nsec)/1000.0f;
  // cout<<time_diff(time1,time2).tv_sec<<":"<<time_diff(time1,time2).tv_nsec<<endl;
  if (time_diff(time1,time2).tv_nsec > fr) {
    if (togSimulate) {
      // writeCpy(0,1,animFrameNum);
      animFrameNum++;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    glutPostRedisplay(); // causes draw to loop forever
  }

  // glutPostRedisplay(); // causes draw to loop forever

}

///////////////////////////////////////////////////////////////////////////////
// Misc functions
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
  sdkDeleteTimer(&timer);
  glDeleteBuffers(1, &bufferObj);
}

void timerEvent(int value) {
  // glutPostRedisplay();
  glutTimerFunc(REFRESH_DELAY, timerEvent,0);
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
    case 'c':
        initArrays();
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
    case ']':
        diff += 0.000001f;
        // if (diff >= 1.) diff = 1.;
        printf("Diff: %f\n", diff);
        break;
    case '[':
        diff -= 0.000001f;
        if (diff <= 0.) diff = 0.;
        printf("Diff: %f\n", diff);
        break;
    case '1':
        drawChem = 0;
        break;
    case '2':
        drawChem = 1;
        break;
    case '0':
        visc += 0.000001f;
        // if (visc >= 1.) visc = 1.;
        printf("Visc: %f\n", visc);
        break;
    case '9':
        visc -= 0.000001f;
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

void mouse_func ( int button, int state, int x, int y ) {
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
  glutTimerFunc(REFRESH_DELAY, timerEvent,0);
  glutMainLoop();

  cudaDeviceReset();

}
