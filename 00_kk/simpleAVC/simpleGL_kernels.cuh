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

__global__ void getSum( float *_data, float _sum ) {
  int x = getX();
  int y = getY();

  _sum += _data[IX(x,y)];
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
  if ( posX < .72 && posX > .45 && posY < .51 && posY > .495 ) {
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

  // if (x>0 && x<DIM-1 && y>0 && y<DIM-1){
    field[id] = (float)(field0[id] + ((float)a*(field[IX(x-1,y)] + field[IX(x+1,y)] + field[IX(x,y-1)] + field[IX(x,y+1)]))) / c;
  // }
}

/**
 * Calculate the buoyancy force as part of the velocity solver.
 * Fbuoy = -a*d*Y + b*(T-Tamb)*Y where Y = (0,1). The constants
 * a and b are positive with appropriate (physically meaningful)
 * units. T is the temperature at the current cell, Tamb is the
 * average temperature of the fluid grid. The density d provides
 * a mass that counteracts the buoyancy force.
 *
 * In this simplified implementation, we say that the tempterature
 * is synonymous with density (since smoke is *hot*) and because
 * there are no other heat sources we can just use the density
 * field instead of a new, seperate temperature field.
 *
 * @param Fbuoy Array to store buoyancy force for each cell.
 **/

__global__ void buoyancy(float *Fbuoy, float *dens, float _Tamb, float Y)
{
  int x = getX();
  int y = getY();
  int id = IX(x,y);
  float a = 0.000625f;
  // float b = 0.025f;
  float b = 0.01f;
  Fbuoy[id] = a * dens[id] + -b * (dens[id] - _Tamb) * Y;
}


/**
 * Calculate the curl at position (i, j) in the fluid grid.
 * Physically this represents the vortex strength at the
 * cell. Computed as follows: w = (del x U) where U is the
 * velocity vector at (i, j).
 *
 * @param i The x index of the cell.
 * @param j The y index of the cell.
 **/

__device__ float curl(int i, int j, float *u, float *v)
{
  float du_dy = (u[IX(i, j+1)] - u[IX(i, j-1)]) * 0.5f;
  float dv_dx = (v[IX(i+1, j)] - v[IX(i-1, j)]) * 0.5f;

  // return du_dy - dv_dx;
  return du_dy - dv_dx;
}


/**
 * Calculate the vorticity confinement force for each cell
 * in the fluid grid. At a point (i,j), Fvc = N x w where
 * w is the curl at (i,j) and N = del |w| / |del |w||.
 * N is the vector pointing to the vortex center, hence we
 * add force perpendicular to N.
 *
 * @param Fvc_x The array to store the x component of the
 *        vorticity confinement force for each cell.
 * @param Fvc_y The array to store the y component of the
 *        vorticity confinement force for each cell.
 **/

__global__ void vorticityConfinement(float *Fvc_x, float *Fvc_y, float *u, float *v)
{
  int x = getX();
  int y = getY();
  int id = IX(x,y);

  float dw_dx, dw_dy;
  float length;
  float vel;

    if (x>0 && x<DIM-1 && y>0 && y<DIM-1){
    // Calculate magnitude of curl(u,v) for each cell. (|w|)
    // curl[I(i, j)] = Math.abs(curl(i, j));

      // Find derivative of the magnitude (n = del |w|)
      dw_dx = ( abs(curl(x+1,y, u, v)) - abs(curl(x-1,y, u, v)) ) * 0.5f;
      dw_dy = ( abs(curl(x,y+1, u, v)) - abs(curl(x,y-1, u, v)) ) * 0.5f;

      // Calculate vector length. (|n|)
      // Add small factor to prevent divide by zeros.
      length = sqrt(dw_dx * dw_dx + dw_dy * dw_dy);
      if (length == 0.0) length -= 0.000001f;
      // N = ( n/|n| )
      dw_dx /= length;
      dw_dy /= length;

      vel = curl(x, y, u, v);

      // N x w
      Fvc_x[id] = dw_dy * -vel;
      Fvc_y[id] = dw_dx *  vel;
    }

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

  // float F = 0.05;
  // float k = 0.0675;
  float F = 0.05;
  float k = 0.06;
  float A = _chemA[id];
  float B = _chemB[id];
  // float rA = A;
  // float rB = B;

  float reactionA = -A * (B*B) + (F * (1.0-A));
  float reactionB = A * (B*B) - (F+k)*B;
  _chemA[id] += (dt * reactionA);
  _chemB[id] += (dt * reactionB);
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
