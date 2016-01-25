#include <stdio.h>
#include <stdlib.h>
#include "writeData.h"
#include "simpleGL_kernels.cuh"

#define MAX(a,b) ((a > b) ? a : b)
#define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}
#define REFRESH_DELAY     10 //ms

dim3 grid, threads;

int size = 0;
int win_x = 512;
int win_y = 512;
int internalFormat = 4;
float dt = 0.1;
float diff = 0.0001f;
float visc = 0.000f;
float force = 1.0;
float source_density = 100.0;

GLuint  bufferObj, bufferObj2;
GLuint  textureID;
GLuint FramebufferName, rndrTx, rndrDepthTx;
cudaGraphicsResource_t resource1;

float avgFPS = 0.0f;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int frameNum = 0;
int animFrameNum = 0;
float framerate_sec = 1.0f/60.0f;

StopWatchInterface *timer = NULL;
timespec time1, time2;
timespec time_diff(timespec start, timespec end);

float *u, *v, *u_prev, *v_prev, *source, *dens, *dens_prev;
float4 *displayPtr, *rndrTx_ptr;
float *h_u, *h_v, *h_d;

bool hasRunOnce = false;
bool writeData = false;

// mouse controls
static int mouse_down[3];
int mouse_x, mouse_y, mouse_x_old, mouse_y_old;
bool togSimulate = false;
int max_simulate = 0;
bool togDensity = true;
bool togVelocity = true;

int ID(int i, int j) { return (i+((N+2)*j)); }

// Convert to webm:
// png2yuv -I p -f 60 -b 1 -n 1628 -j cuda_x%05d.png > cuda_YUV.yuv
// vpxenc --good --cpu-used=0 --auto-alt-ref=1 --lag-in-frames=16 --end-usage=vbr --passes=2 --threads=2 --target-bitrate=3000 -o cuda_WEBM.webm cuda_YUV.yuv
///////////////////////////////////////////////////////////////////////////////
// Initialize Variables
///////////////////////////////////////////////////////////////////////////////
void initVariables(int argc, char *argv[]) {
  // grid = dim3(DIM/16,DIM/16);
  // threads = dim3(16,16);

  threads = dim3(16,16);
  grid.x = (DIM + threads.x - 1) / threads.x;
  grid.y = (DIM + threads.y - 1) / threads.y;

  size = (N+2)*(N+2);
  displayPtr = (float4*)malloc(sizeof(float4)*DIM*DIM);
  rndrTx_ptr = (float4*)malloc(sizeof(float4)*win_x*win_y);
  h_u = new float[size];
  h_v = new float[size];
  h_d = new float[size];
  for (int i=0; i<size; i++){
    h_u[i] = 0.0;
    h_v[i] = 0.0;
    h_d[i] = 0.0;
  }

  writeData = argv[1];

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
  glutInitWindowSize ( win_x, win_y );
  glutCreateWindow( "Simple Advection" );
  glewInit();

  // Framebuffer
  glGenFramebuffersEXT(1, &FramebufferName);
  glBindFramebufferEXT(GL_FRAMEBUFFER, FramebufferName);

  // Framebuffer's texture
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &rndrTx);
  glBindTexture(GL_TEXTURE_2D, rndrTx);
  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, win_x, win_y, 0, GL_RGBA, GL_FLOAT, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glBindTexture(GL_TEXTURE_2D, 0);

  // The depth buffer
  glGenRenderbuffersEXT(1, &rndrDepthTx);
  glBindRenderbufferEXT(GL_RENDERBUFFER, rndrDepthTx);
  glRenderbufferStorageEXT(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, win_x, win_y);
  glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rndrDepthTx);
  glFramebufferTextureEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, rndrTx, 0);

  // Set the list of draw buffers.
  GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
  glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

  // Set up CUDA buffer object and bind texture to it
  // Later we register a cuda graphics resource to this in order to write to a texture
  glGenBuffers( 1, &bufferObj );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
  glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, sizeof(float4) * DIM * DIM, NULL, GL_DYNAMIC_DRAW_ARB );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );

  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, DIM, DIM, 0, GL_BGRA, GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  // Clean up
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

  checkCudaErrors(cudaMemcpy(u, h_u, sizeof(float)*size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(v, h_v, sizeof(float)*size, cudaMemcpyHostToDevice));
}

void initArrays() {
  ClearArray<<<grid,threads>>>(u, 0.0);
  ClearArray<<<grid,threads>>>(u_prev, 0.0);
  ClearArray<<<grid,threads>>>(v, 0.0);
  ClearArray<<<grid,threads>>>(v_prev, 0.0);
  ClearArray<<<grid,threads>>>(dens, 0.0);
  ClearArray<<<grid,threads>>>(dens_prev, 0.0);
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
void get_from_UI(float *d, float *u, float *v) {

  int i, j = (N+2)*(N+2);

  // which is faster?
  // ClearThreeArrays<<<grid,threads>>>(d, u, v);
  ClearArray<<<grid,threads>>>(d, 0.0);
  ClearArray<<<grid,threads>>>(u, 0.0);
  ClearArray<<<grid,threads>>>(v, 0.0);

  DrawSquare<<<grid,threads>>>(d);

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
    GetFromUI<<<grid,threads>>>(d, i, j, source_density);
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

void dens_step ( float * field, float * field0, float *u, float *v, float diff, float dt )
{
  AddSource<<<grid,threads>>>( field, field0, dt );
  SWAP ( field0, field );
  diffuse_step( 0, field, field0, diff, dt);

  SWAP ( field0, field );
  advect_step(0, field, field0, u, v, dt);
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
    get_from_UI(dens_prev, u_prev, v_prev);
    vel_step( u, v, u_prev, v_prev, visc, dt );
    dens_step( dens, dens_prev, u, v, diff, dt );
    MakeColor<<<grid,threads>>>(dens, displayPtr);
    // MakeColor<<<grid,threads>>>(u, v, displayPtr);
  }

  size_t  sizeT;
  cudaGraphicsMapResources( 1, &resource1, 0 );
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&displayPtr, &sizeT, resource1));
  checkCudaErrors(cudaGraphicsUnmapResources( 1, &resource1, 0 ));

  sdkStopTimer(&timer);
  computeFPS();
}

///////////////////////////////////////////////////////////////////////////////
// Draw
///////////////////////////////////////////////////////////////////////////////
static void pre_display ( void ) {
  // bind a framebuffer and render everything afterwards into that
  glBindFramebufferEXT(GL_FRAMEBUFFER, FramebufferName);
  glViewport ( 0, 0, win_x, win_y );
  glMatrixMode ( GL_PROJECTION );
  glLoadIdentity ();
  gluOrtho2D ( 0.0, 1.0, 0.0, 1.0 );
  glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
  glClear(GL_COLOR_BUFFER_BIT);
}

static void post_display ( void ) {
  // unbind the framebuffer and draw its texture
  glBindFramebufferEXT(GL_FRAMEBUFFER, 0);

  glColor3f(1,1,1);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, rndrTx);
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

  // now handle looping and writing data
  if (time_diff(time1,time2).tv_nsec > framerate_sec) {
    if (togSimulate) {
      if (writeData) {
        // glReadPixels(0, 0, win_x, win_y, GL_RGBA, GL_FLOAT, rndrTx_ptr);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, rndrTx_ptr);
        writeImage(rndrTx_ptr, animFrameNum, win_x, win_y, internalFormat);
      }
      animFrameNum++;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
      glutPostRedisplay(); // causes draw to loop forever
    }
  }
}

void draw_density() {
  glEnable (GL_TEXTURE_2D);
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
}

void draw_velocity() {
	float h = (float)1.0f/float(N);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaMemcpy(h_u, u, sizeof(float)*size, cudaMemcpyDeviceToHost ));
  checkCudaErrors(cudaMemcpy(h_v, v, sizeof(float)*size, cudaMemcpyDeviceToHost ));

  glDisable(GL_LIGHTING);
  glDisable(GL_TEXTURE_2D);
	glLineWidth ( 1.0f );
  glColor3f ( 1.0f, 1.0f, 1.0f );

	glBegin ( GL_LINES );
		for ( int i=0 ; i<DIM ; i+=2 ) {
			float x = (i-0.5f)*h;
			for ( int j=0 ; j<DIM ; j+=2 ) {
				float y = (j-0.5f)*h;
        int id = ((i)+DIM*(j));
  				glVertex2f ( x, y );
  				glVertex2f ( x+h_u[id], y+h_v[id] );
			}
		}
	glEnd ();
}

static void draw_func( void ) {

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);

  pre_display();

  if (togDensity) draw_density();
  if (togVelocity) draw_velocity();

  post_display();

}

///////////////////////////////////////////////////////////////////////////////
// Misc functions
///////////////////////////////////////
// Close
///////////////////////////////////////
static void FreeResource( void ){
  sdkDeleteTimer(&timer);
  glDeleteBuffers(1, &bufferObj);
}

void timerEvent(int value) {
  glutPostRedisplay();
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
        // writeImage(frameNum);
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
    case 'd':
      togDensity = !togDensity;
      printf("toggle density: %d\n", togDensity);
      break;
    case 'v':
      togVelocity = !togVelocity;
      printf("toggle velocity: %d\n", togVelocity);
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
	mouse_y_old = mouse_y = y;

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
  initVariables(argc, argv);
  initGL(argc, argv);
  initCUDA();


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
