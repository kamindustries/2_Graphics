#define     DIM    256
#define     DT    .1


///////////////////////////////////////
// CUDA Kernel
///////////////////////////////////////
__device__ int checkPosition(int _pos){
  int dmax = DIM*DIM;
  if (_pos < 0){
    // _pos = dmax+_pos;
    _pos += DIM;
    // return _pos % dmax;
    return _pos;
  }
  else return _pos % dmax;
}

__global__ void ClearArray( float *_chem, float value) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  _chem[offset] = value;
}

__global__ void DrawSquare( float *_chem ) {
  if (threadIdx.x > DIM || threadIdx.y > DIM) return;

  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + (y * blockDim.x * gridDim.x);

  // q1. draws a square
  float posX = (float)x/DIM;
  float posY = (float)y/DIM;
  // if ( x < 140 && x > 116 && y < 140 && y > 116 ) {
  if ( x < 200 && x > 116 && y < 140 && y > 30 ) {
  // if ( posX < .75 && posX > .45 && posY < .55 && posY > .45 ) {
  // if ( posX < m_x+.05 && posX > m_x-.05 && posY < m_y+.05 && posY > m_y-.05 ) {    //use mouse position
    _chem[offset] = 1.0;
  }

}

__global__ void Diffusion( float *_chem, float *_lap, float _difConst, int mouse_x, int mouse_y) {
  if (threadIdx.x > DIM || threadIdx.y > DIM) return;

  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + (y * blockDim.x * gridDim.x);

  // constants
  // float xLength = (float)DIM/100.0;
  float xLength = 2.56f;
  // float dx = (float)xLength/DIM;
  float dx = 0.01f;
  float alpha = (float)(_difConst * .1f / (float)(dx*dx));

  // int n1 = (x+1)%DIM;
  // int n2 = (x-1)%DIM;
  // int n3 = (y+1)%DIM;
  // int n4 = (y-1)%DIM;

  // if (n2 < 0) n2 += DIM;
  // if (n4 < 0) n4 += DIM;

  // n1 = ((n1 + y * blockDim.x * gridDim.x)) % (DIM*DIM);
  // n2 = ((n2 + y * blockDim.x * gridDim.x)) % (DIM*DIM);
  // n3 = ((x + n3 * blockDim.x * gridDim.x)) % (DIM*DIM);
  // n4 = ((x + n4 * blockDim.x * gridDim.x)) % (DIM*DIM);
  // if (n1 > (y*DIM)+DIM) n1 -= DIM;

  int n1 = offset + 1;
  int n2 = offset - 1;
  int n3 = offset + DIM;
  int n4 = offset - DIM;

  if (n1 > ((DIM-1) + (y * blockDim.x * gridDim.x))) n1 -= DIM;
  if (n1 >= DIM*DIM) n1 -= DIM;

  if (n2 < (0 + (y * blockDim.x * gridDim.x))) n2 = ((DIM-1) + (y * blockDim.x * gridDim.x));
  if (n2 < 0) n2 += DIM;

  if (n3 >= DIM*DIM) n3 = x;

  if (n4 < 0) n4 = (DIM*DIM) - DIM + x;

  // int n1 = checkPosition((x+1) + y * blockDim.x * gridDim.x);
  // int n2 = checkPosition((x-1) + y * blockDim.x * gridDim.x);
  // int n3 = checkPosition(x + (y+1) * blockDim.x * gridDim.x);
  // int n4 = checkPosition(x + (y-1) * blockDim.x * gridDim.x);
  // __syncthreads();

  _lap[offset] = (float)(-4.0f * _chem[offset]) + (float)(_chem[n1] + _chem[n2] + _chem[n3] + _chem[n4]);
  _lap[offset] = (float)_lap[offset]*alpha;

}

__global__ void AddLaplacian( float *_chem, float *_lap) {
  if (threadIdx.x > DIM || threadIdx.y > DIM) return;

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  _chem[offset] += _lap[offset];
}

__global__ void React( float *_chemA, float *_chemB) {
  // if (threadIdx.x > DIM || threadIdx.y > DIM) return;

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float F = 0.05f;
  float k = 0.0675f;
  float A = float(_chemA[offset]);
  float B = float(_chemB[offset]);

  float reactionA = (float)(-A * (float)(B*B) + (float)(F * (float)(1.0f-A)));
  float reactionB = (float)(A * (float)(B*B) - (float)(F+k)*B);

  _chemA[offset] += (float)(.1f * reactionA);
  _chemB[offset] += (float)(.1f * reactionB);
}

__global__ void MakeColor( float *field, float4 *displayPtr) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int id = x + y * blockDim.x * gridDim.x;

  displayPtr[id] = make_float4(field[id],field[id],field[id],1.0);
}
