///////////////////////////////////////////////////////////////////////////////
// CUDA Kernels
///////////////////////////////////////////////////////////////////////////////

#include <GL/glew.h>
// #include <GL/freeglut.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <timer.h>               // timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <helper_math.h>

#define     DIM    512
#define     N    DIM-2

// Get 1d index from 2d coords
//
__device__ int IX( int x, int y) {
  if (x >= DIM) x = 0;
  if (x < 0) x = DIM-1;
  if (y >= DIM) y = 0;
  if (y < 0) y = DIM-1;
  return x + (y * blockDim.x * gridDim.x);
}

__device__ int getX() {
  return threadIdx.x + (blockIdx.x * blockDim.x);
}

__device__ int getY() {
  return threadIdx.y + (blockIdx.y * blockDim.y);
}

// Set boundary conditions
__device__ void set_bnd( int b, int x, int y, float *field) {
  int sz = DIM*DIM;
  int id = IX(x,y);

  if (x==0)       field[id] = b==1 ? -1*field[IX(1,y)] : field[IX(1,y)];
  if (x==DIM-1)   field[id] = b==1 ? -1*field[IX(DIM-2,y)] : field[IX(DIM-2,y)];
  if (y==0)       field[id] = b==2 ? -1*field[IX(x,1)] : field[IX(x,1)];
  if (y==DIM-1)   field[id] = b==2 ? -1*field[IX(x,DIM-2)] : field[IX(x,DIM-2)];

  if (id == 0)      field[id] = 0.5*(field[IX(1,0)]+field[IX(0,1)]);  // southwest
  if (id == sz-DIM) field[id] = 0.5*(field[IX(1,DIM-1)]+field[IX(0, DIM-2)]); // northwest
  if (id == DIM-1)  field[id] = 0.5*(field[IX(DIM-2,0)]+field[IX(DIM-1,1)]); // southeast
  if (id == sz-1)   field[id] = 0.5*(field[IX(DIM-2,DIM-1)]+field[IX(DIM-1,DIM-2)]); // northeast
}

__global__ void SetBoundary( int b, float *field ) {
  int x = getX();
  int y = getY();

  set_bnd(b, x, y, field);
}

// Draw a square
//
__global__ void DrawSquare( float *field, float value ) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  // q1. draws a square
  float posX = (float)x/DIM;
  float posY = (float)y/DIM;
  if ( posX < .75 && posX > .45 && posY < .51 && posY > .48 ) {
    field[id] = value;
  }
}

__global__ void ClearArray( float4 *field, float value ) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  field[id] = make_float4(value,value,value,1.);
}

__global__ void ClearArray( float *field, float value ) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  field[id] = value;
}

__global__ void GetFromUI ( float * field, int x_coord, int y_coord, float value ) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  if (x>x_coord-2 && x<x_coord+2 && y>y_coord-2 && y<y_coord+2){
  // if (x == x_coord && y==y_coord){
    field[id] = value;
  }
  else return;
}

// __global__ void ClearThreeArrays ( float * d, float * u, float * v ) {
//   int x = getX();
//   int y = getY();
//   int id = IX(x,y);
//
//   u[id] = v[id] = d[id] = 0.0;
// }

__global__ void InitVelocity ( float * field ) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);
  float s = sin((float(x)/float(N)) * 3.1459 * 4);
  field[id] += s;
}

__global__ void AddStaticVelocity ( float * field, float value, float dt ) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  float dF = float(DIM);
  float yF = float(y);
  float i0 = abs(0.5 - (yF/dF)) * -2.0;
  i0 = i0+1.0;

  // i0 = yF/dF;
  // if (y > .4 && y < .6) {
    // field[id] += (i0 * dt) * value;
    field[id] += sin(((yF/dF) * 3.14159 * value))*dt;
  // }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

__global__ void AddSource ( float *field, float *source, float dt ) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  field[id] += (dt * source[id]);
}

__global__ void LinSolve( float *field, float *field0, float a, float c) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  field[id] = (float)(field0[id] + ((float)a*(field[IX(x-1,y)] + field[IX(x+1,y)] + field[IX(x,y-1)] + field[IX(x,y+1)]))) / c;
}

__global__ void Advect ( float *field, float *field0, float *u, float *v, float dt ) {
  int i = getX();
  int j = getY();
  int id = IX(i,j);

  int i0, j0, i1, j1;
  float x, y, s0, t0, s1, t1, dt0;

  dt0 = (float)dt*float(N);

  // if (x>0 && x<DIM-1 && y>0 && y<DIM-1){
    x = (float)i - dt0 * u[id];
    y = (float)j - dt0 * v[id];

    if (x < 0.5f) x = 0.5f;
    if (x > (float)N+0.5f) x = (float)N+0.5f;
    i0 = (int)x;
    i1 = i0+1;

    if (y < 0.5f) y = 0.5f;
    if (y > (float)N+0.5f) y = (float)N+0.5f;
    j0 = (int)y;
    j1 = j0+1;

    s1 = (float)x-i0;
    s0 = (float)1-s1;
    t1 = (float)y-j0;
    t0 = (float)1-t1;

    field[id] = (float)s0*(t0*field0[IX(i0,j0)] + t1*field0[IX(i0,j1)])+
			 				         s1*(t0*field0[IX(i1,j0)] + t1*field0[IX(i1,j1)]);
  // }
}

__global__ void Project ( float *u, float *v, float *p, float *div ) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  if (x>0 && x<DIM-1 && y>0 && y<DIM-1){
    div[id] = -0.5 *(u[IX(x+1,y)] - u[IX(x-1,y)] + v[IX(x,y+1)] - v[IX(x,y-1)]) / float(N);
    p[id] = 0;
  }
}

__global__ void ProjectFinish ( float *u, float *v, float *p, float *div ) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  if (x>0 && x<DIM-1 && y>0 && y<DIM-1){
    u[id] -= (0.5 * float(N) * (p[IX(x+1,y)] - p[IX(x-1,y)]));
    v[id] -= (0.5 * float(N) * (p[IX(x,y+1)] - p[IX(x,y-1)]));
  }
}

__global__ void Diffusion( float *_chem, float *_lap, float _difConst, float dt) {
  if (threadIdx.x > DIM || threadIdx.y > DIM) return;

  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + (y * blockDim.x * gridDim.x);

  if (x>0 && x<DIM-1 && y>0 && y<DIM-1){
    int n1 = offset + 1;
    int n2 = offset - 1;
    int n3 = offset + DIM;
    int n4 = offset - DIM;

    // constants
    // float xLength = (float)DIM/100.0;
    float xLength = 2.56f;
    // float dx = (float)xLength/DIM;
    float dx = 0.01f;
    float alpha = (float)(_difConst * dt / (float)(dx*dx));

    _lap[offset] = (float)(-4.0f * _chem[offset]) + (float)(_chem[n1] + _chem[n2] + _chem[n3] + _chem[n4]);
    _lap[offset] = (float)_lap[offset]*alpha;
  }
}

__global__ void AddLaplacian( float *_chem, float *_lap) {
  if (threadIdx.x > DIM || threadIdx.y > DIM) return;

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  _chem[offset] += _lap[offset];
}

__global__ void React( float *_chemA, float *_chemB, float dt) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  float F = 0.05;
  float k = 0.0675;
  float A = _chemA[id];
  float B = _chemB[id];
  // float rA = A;
  // float rB = B;

  float reactionA = -A * (B*B) + (F * (1.0-A));
  float reactionB = A * (B*B) - (F+k)*B;
  _chemA[id] += (dt * reactionA);
  _chemB[id] += (dt * reactionB);
}

__global__ void ReactStable( float *_chemA, float *_chemB, float dt) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  float F = 0.05;
  float k = 0.0675;
  float A = _chemA[id];
  float B = _chemB[id];
  float B_new = B;
  float B_og = B;
  float dfdx = B;
  float f = 0.0;

  for (int i = 0; i < 100; i++) {
    B = B_new;
    f = B_og + dt*( ((A + F*dt)/(dt*(B*B) + F*dt + 1.0f))*(B*B) - (F+k)*B );
    float numerator = (2.0f*A*B*F*(dt*dt)) + (2.0f*A*B*dt) + (2.0f*(F*F)*B*(dt*dt*dt)) + (2.0f*F*B*(dt*dt));
    float denominator = ((B*B*B*B)*(dt*dt) + (2.0f*((B*B)*F*(dt*dt))) + (B*B) + ((F*F)*(dt*dt)) + (2.0f*(F*dt)) + ((B*B)*dt)) + 1.0f;
    dfdx = (numerator / denominator) - (F*dt) + (k*dt);
    // Newton-Raphson step
    B_new = B - (f/dfdx);
    if (abs(B-B_new) < 1e-6) {
      B = B_new;
      A = (A + F*dt) / (dt*(B*B) + F*dt + 1.0f);
      _chemA[id] = A;
      _chemB[id] = B;
      return;
    }
  }

  A = (A + F*dt) / (dt*(B*B) + F*dt + 1.0f);

  _chemA[id] = A;
  _chemB[id] = B;
}

__global__ void DiffusionTed( float *field, float *field0, float dt, float a, float h ) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  float q = (dt*a)/(h*h);
  field[id] = field0[id] + q*(field[IX(x-1,y)] + field[IX(x,y-1)] - 4*field[id] +
                              field[IX(x+1,y)] + field[IX(x,y+1)]);
}

// really dont like that i have to do this...
//
__global__ void MakeColor( float *data, float4 *_toDisplay) {
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  float Cd = data[id];
  _toDisplay[id] = make_float4(Cd, Cd, Cd, 1.0);
}
